import math
import time
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

import numpy as np
import pandas as pd

from sequence.components.optical_channel import QuantumChannel, ClassicalChannel, EveQuantumChannel
from sequence.components.thermal_noise_source import ThermalNoiseSource, compute_n_B, _c
from sequence.kernel.event import Event
from sequence.kernel.process import Process
from sequence.kernel.timeline import Timeline
from sequence.qkd.BB84 import pair_bb84_protocols
from sequence.qkd.B92 import pair_b92_protocols
from sequence.qkd.COW import pair_cow_protocols
from sequence.topology.node import QKDNode, EveNode
from sequence.utils.encoding_cow import time_bin_cow
import sequence.utils.log as log

from QCLoss.loss import channel_FSO_loss, cn2, polarization_fidelity, phase_noise, make_atmospheric_phase_process, wind_speed_perp
from scenarios import materialize


# ═══════════════════════════════════════════════════════════════════════
#  Protocol registry — every per-protocol difference lives here
# ═══════════════════════════════════════════════════════════════════════
PROTOCOL_REGISTRY = {
    "BB84": dict(qkdtype=0, pair_fn=pair_bb84_protocols, encoding=None,
        log_module="BB84", has_eve=False, needs_visibility=False,
        ls_key="ls_params", det_key="detector_params", source_type="sps",
    ),
    "B92": dict(qkdtype=1, pair_fn=pair_b92_protocols, encoding=None,
        log_module="B92", has_eve=False, needs_visibility=False,
        ls_key="ls_params", det_key="detector_params", source_type="sps",
    ),
    "COW": dict(qkdtype=2, pair_fn=pair_cow_protocols, encoding=time_bin_cow,
        log_module="COW", has_eve=False, needs_visibility=True,
        ls_key="ls_params_cow", det_key="detector_params_cow", source_type="wcp",
    ),
    "BB84+Eve": dict(qkdtype=0, pair_fn=pair_bb84_protocols, encoding=None,
        log_module="BB84", has_eve=True, needs_visibility=False,
        ls_key="ls_params", det_key="detector_params", source_type="sps",
    ),
    "B92+Eve": dict(qkdtype=1, pair_fn=pair_b92_protocols, encoding=None,
        log_module="B92", has_eve=True, needs_visibility=False,
        ls_key="ls_params", det_key="detector_params", source_type="sps",
    ),
    "COW+Eve": dict(qkdtype=2, pair_fn=pair_cow_protocols, encoding=time_bin_cow,
        log_module="COW", has_eve=True, needs_visibility=True,
        ls_key="ls_params_cow", det_key="detector_params_cow", source_type="wcp",
    ),
}

# Metric column names used in the output CSVs (label, dict-key)
_METRIC_COLS = [
    ("R_sk", "skr"), ("QBER", "qber"), ("Throughputs", "throughputs"),
    ("Latency", "latency"), ("Loss", "loss"), ("R_s", "rs"),
]
_VIS_COL = ("Visibility", "visibility")


# ═══════════════════════════════════════════════════════════════════════
#  Metric helpers
# ═══════════════════════════════════════════════════════════════════════
def binary_entropy(Q):
    Q = max(0.0, min(1.0, Q))
    if Q == 0 or Q == 1:
        return 0
    return -Q * math.log2(Q) - (1 - Q) * math.log2(1 - Q)


def _safe_mean(lst, default=np.nan):
    """np.nanmean(lst) but tolerant of None / empty / scalar inputs."""
    if lst is None:
        return default
    try:
        if len(lst) == 0:
            return default
    except TypeError:                       # already a scalar
        return float(lst)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = float(np.nanmean(lst))
    return result if not np.isnan(result) else default


def _collect_metrics(protocol):
    """Generic metrics: QBER, throughput, latency, SKR, loss, R_s."""
    qber_list = protocol.error_rates
    throughputs = (np.mean(protocol.throughputs) if len(protocol.throughputs) > 0 else 0.0)
    latency = protocol.latency

    if not qber_list or protocol.send_bits_length == 0:
        return qber_list, throughputs, latency, 0.0, 0.0

    rs_list, skr_sum = [], 0.0
    for i, e in enumerate(qber_list):
        rs = protocol.sifted_bits_length[i] / protocol.send_bits_length
        skr_sum += max(0.0, rs * (1 - 2 * binary_entropy(e)))
        rs_list.append(rs)
    return qber_list, throughputs, latency, skr_sum / len(qber_list), float(np.mean(rs_list))


def _collect_cow_metrics(protocol, visibility, ls_params, loss):
    """COW metrics with visibility-adjusted SKR (DOI 10.1063/1.2126792)."""
    qber_list = protocol.error_rates
    throughputs = (np.mean(protocol.throughputs) if len(protocol.throughputs) > 0 else 0.0)
    latency = protocol.latency

    if not qber_list or protocol.send_bits_length == 0:
        return qber_list, throughputs, latency, 0.0, 0.0

    mu = ls_params["mean_photon_num"]
    t = 1 - loss
    r = mu * (1 - t)
    rs_list, skr_sum = [], 0.0
    for i, e in enumerate(qber_list):
        rs = protocol.sifted_bits_length[i] / protocol.send_bits_length
        v = visibility[i]
        eve_info = r + ((1 - v) * (1 + math.exp(-mu * t)) / (2 * math.exp(-mu * t)))
        skr_sum += max(0.0, rs * (1 - binary_entropy(e) - eve_info))
        rs_list.append(rs)
    return qber_list, throughputs, latency, skr_sum / len(qber_list), float(np.mean(rs_list))


def _attach_thermal_noise(tl, detector, ls_params, thermal_params):
    """Create and connect a ThermalNoiseSource to `detector`."""
    n_B = compute_n_B(
        wavelength_nm=ls_params["wavelength"],
        delta_lambda_nm=thermal_params["delta_lambda_nm"],
        delta_t_ns=thermal_params["delta_t_ns"],
        omega_fov_sr=thermal_params["omega_fov_sr"],
        a_R_cm=thermal_params["a_R_cm"],
        B_sky=thermal_params["B_sky"],
    )
    encoding = detector.owner.encoding if hasattr(detector.owner, "encoding") else None
    src = ThermalNoiseSource(name=f"thermal_{detector.name}", timeline=tl, n_B=n_B, frequency=ls_params["frequency"], encoding_type=encoding)
    src.add_receiver(detector)
    tl.entities[src.name] = src
    return src


# ═══════════════════════════════════════════════════════════════════════
#  Building blocks for the simulation runner
# ═══════════════════════════════════════════════════════════════════════
def _setup_atm_processes(is_cow, has_eve, loss_parameters, ls_params, distance, eve_position, stop_time_ps):
    """Pre-generate atmospheric piston processes for one simulation run.

    Returns:
        (atm_ab, atm_eb): atmospheric phase processes for segments
        Alice -> (Bob or Eve) and Eve -> Bob. ``atm_eb`` is ``None``
        when no Eve is present, and both are ``None`` for non-COW
        protocols or when ``loss_parameters`` is unavailable.
    """
    if not is_cow or loss_parameters is None:
        return None, None

    if has_eve:
        d1 = distance * eve_position
        d2 = distance * (1.0 - eve_position)
        atm_ab = make_atmospheric_phase_process(distance=d1, timeline_stop_time_ps=stop_time_ps, ls_params=ls_params, loss_parameters=loss_parameters, seed=3)
        atm_eb = make_atmospheric_phase_process(distance=d2, timeline_stop_time_ps=stop_time_ps, ls_params=ls_params, loss_parameters=loss_parameters, seed=4)
        return atm_ab, atm_eb

    atm_ab = make_atmospheric_phase_process(distance=distance, timeline_stop_time_ps=stop_time_ps, ls_params=ls_params, loss_parameters=loss_parameters, seed=3)
    return atm_ab, None


def _make_qchannels(tl, cfg, distance, attenuation, polarization_fidelity,
                    loss, atm_ab, atm_eb, phase_noise_coefficient,
                    eve_intercept_rate, eve_position):
    """Build qc0 (with or without Eve) and qc1 with consistent kwargs.

    ``phase_noise_coefficient`` is always forwarded -- it is ignored at
    runtime for non-time_bin_cow encodings (see
    ``QuantumChannel._apply_channel_noise``), so it's safe to pass it
    unconditionally and avoids inconsistent values between segments.
    """
    qc_kwargs = dict(
        distance=distance,
        polarization_fidelity=polarization_fidelity,
        attenuation=attenuation, loss=loss,
        atmospheric_phase_process=atm_ab,
        phase_noise_coefficient=phase_noise_coefficient,
    )

    if cfg["has_eve"]:
        eve_kwargs = dict(intercept_rate=eve_intercept_rate, seed=2)
        if cfg["encoding"] is not None:
            eve_kwargs["encoding"] = cfg["encoding"]
        eve = EveNode("eve", tl, **eve_kwargs)
        qc0 = EveQuantumChannel("qc0", tl, eve_node=eve, eve_position=eve_position,
            atmospheric_phase_process_seg2=atm_eb,
            phase_noise_coefficient_seg2=phase_noise_coefficient,
            **qc_kwargs,
        )
    else:
        qc0 = QuantumChannel("qc0", tl, **qc_kwargs)
    qc1 = QuantumChannel("qc1", tl, **qc_kwargs)
    return qc0, qc1


def _configure_bob_cow_interferometer(bob, interferometer_phase_error):
    """Set the Michelson phase error on Bob's QSDetectorCOW (if present)."""
    from sequence.components.qsdetector_cow import QSDetectorCOW
    for comp in bob.components.values():
        if isinstance(comp, QSDetectorCOW):
            comp.interferometer.phase_error = interferometer_phase_error
            return


# ═══════════════════════════════════════════════════════════════════════
#  Unified simulation runner
# ═══════════════════════════════════════════════════════════════════════
def run_qkd_simulation(
    protocol, ls_params, detector_params, *,
    runtime=20, log_filename=-1, distance=1e3,
    polarization_fidelity=0.97, attenuation=2e-4,
    keysize=256, key_num=math.inf,
    source_type=None, loss=None, thermal_params=None,
    phase_noise_coefficient=0.01, interferometer_phase_error=0.20,
    eve_intercept_rate=0.9, eve_position=0.5, loss_parameters=None,
):
    """Run any QKD protocol registered in PROTOCOL_REGISTRY.

    Returns a dict with keys: qber, throughputs, latency, skr, loss, rs,
    plus 'visibility' for COW protocols.
    """
    cfg = PROTOCOL_REGISTRY[protocol]
    is_cow = cfg["qkdtype"] == 2
    src_type = source_type if source_type is not None else cfg["source_type"]

    tl = Timeline(runtime * 1e9)
    tl.show_progress = False

    if log_filename != -1:
        log.set_logger(__name__, tl, log_filename)
        log.set_logger_level("DEBUG")
        log.track_module(cfg["log_module"])
        log.track_module("light_source")

    # Atmospheric piston pre-generation (only meaningful for COW).
    atm_ab, atm_eb = _setup_atm_processes(is_cow, cfg["has_eve"], loss_parameters, ls_params, distance, eve_position, tl.stop_time)

    # Quantum channels (qc0 carries Eve when applicable).
    qc0, qc1 = _make_qchannels(tl, cfg, distance, attenuation, polarization_fidelity, loss, atm_ab, atm_eb, phase_noise_coefficient, eve_intercept_rate, eve_position)

    # Classical channels.
    cc0 = ClassicalChannel("cc0", tl, distance=distance)
    cc1 = ClassicalChannel("cc1", tl, distance=distance)
    cc0.delay += 1e9
    cc1.delay += 1e9

    # Nodes.
    node_kwargs = dict(stack_size=1, qkdtype=cfg["qkdtype"], source_type=src_type)
    if cfg["encoding"] is not None:
        node_kwargs["encoding"] = cfg["encoding"]

    alice = QKDNode("alice", tl, **node_kwargs)
    alice.set_seed(0)
    for k, v in ls_params.items():
        alice.update_lightsource_params(k, v)

    bob = QKDNode("bob", tl, **node_kwargs)
    bob.set_seed(1)
    for i, dp in enumerate(detector_params):
        for k, v in dp.items():
            bob.update_detector_params(i, k, v)

    if is_cow:
        _configure_bob_cow_interferometer(bob, interferometer_phase_error)

    qc0.set_ends(alice, bob.name)
    qc1.set_ends(bob, alice.name)
    cc0.set_ends(alice, bob.name)
    cc1.set_ends(bob, alice.name)

    cfg["pair_fn"](alice.protocol_stack[0], bob.protocol_stack[0])
    tl.schedule(Event(0, Process(alice.protocol_stack[0],
                                 "push", [keysize, key_num, 6e12])))

    tl.init()
    if thermal_params is not None:
        bob_det = bob.components[bob.first_component_name].detectors[0]
        _attach_thermal_noise(tl, bob_det, ls_params, thermal_params)
    tl.run()

    proto_obj = alice.protocol_stack[0]
    if cfg["needs_visibility"]:
        vis = proto_obj.visibility
        qber, th, lat, skr, rs = _collect_cow_metrics(
            proto_obj, vis, ls_params, loss)
        return dict(qber=qber, throughputs=th, latency=lat, skr=skr, loss=loss, rs=rs, visibility=vis)
    qber, th, lat, skr, rs = _collect_metrics(proto_obj)
    return dict(qber=qber, throughputs=th, latency=lat, skr=skr, loss=loss, rs=rs)


# ═══════════════════════════════════════════════════════════════════════
#  Backwards-compatible thin wrappers (preserve original tuple API)
# ═══════════════════════════════════════════════════════════════════════
def _unpack(res):
    base = (res["qber"], res["throughputs"], res["latency"], res["skr"], res["loss"], res["rs"])
    return base + (res["visibility"],) if "visibility" in res else base


def simulation_BB84(*a, **kw):     return _unpack(run_qkd_simulation("BB84", *a, **kw))
def simulation_B92(*a, **kw):      return _unpack(run_qkd_simulation("B92", *a, **kw))
def simulation_COW(*a, **kw):      return _unpack(run_qkd_simulation("COW", *a, **kw))
def simulation_BB84_Eve(*a, **kw): return _unpack(run_qkd_simulation("BB84+Eve", *a, **kw))
def simulation_B92_Eve(*a, **kw):  return _unpack(run_qkd_simulation("B92+Eve", *a, **kw))
def simulation_COW_Eve(*a, **kw):  return _unpack(run_qkd_simulation("COW+Eve", *a, **kw))


# ═══════════════════════════════════════════════════════════════════════
#  Per-task parameter computation
# ═══════════════════════════════════════════════════════════════════════
def _compute_loss(distance, ls_params, loss_parameters):
    """Forward the entire `loss_parameters` dict to channel_FSO_loss."""
    l_p = loss_parameters.copy()
    del l_p["wind_speed_perp"]
    return channel_FSO_loss(distance=distance, wavelength=ls_params["wavelength"], **l_p)


def _compute_polarization_fidelity(distance, ls_params, detector_params, loss_parameters):
    """Compute the polarization fidelity for the given (distance, atmospheric)
    configuration. `detector_params` is the list-of-dicts used elsewhere;
    the count_rate of the first detector defines the detection window."""
    return polarization_fidelity(
        distance=distance,
        wavelength=ls_params["wavelength"],
        w_0=loss_parameters["w_0"],
        photon_number=ls_params["mean_photon_num"],
        pulse_duration=1 / ls_params["frequency"],
        detection_time=detector_params[0]["time_resolution"] * 1e-12,
        friction_velocity=loss_parameters["friction_velocity"],
        height=loss_parameters["height"],
        C_n2=loss_parameters["C_n2"],
    )


def _compute_phase_noise_coefficient(ls_params, loss_parameters):
    """Return phase-noise coefficient in rad/sqrt(m) for the given channel.
    L_0 is derived from `friction_velocity`/`height` via the surface-layer
    model inside `phase_noise()`."""
    return phase_noise(
        wavelength=ls_params["wavelength"],
        C_n2=loss_parameters["C_n2"],
        friction_velocity=loss_parameters["friction_velocity"],
        height=loss_parameters["height"],
    )


def _build_task_kwargs(*, distance, keysize, ls_params, detector_params,
                       loss_parameters, thermal_params, runtime, attenuation,
                       pfid_initial, key_num, source_type, extra_kwargs=None):
    """Single source of truth for one ``run_qkd_simulation`` kwargs dict.

    All derived quantities (loss, polarisation fidelity, phase-noise
    coefficient) are computed here from the atmospheric parameters
    actually used by this task, so different sweep types (variable vs.
    scenario) automatically get them right.
    """
    loss = _compute_loss(distance, ls_params, loss_parameters)
    pfid = (pfid_initial if pfid_initial is not None
            else _compute_polarization_fidelity(
                distance, ls_params, detector_params, loss_parameters))
    pnc = _compute_phase_noise_coefficient(ls_params, loss_parameters)

    kwargs = dict(
        runtime=runtime, distance=distance,
        polarization_fidelity=pfid, attenuation=attenuation,
        keysize=keysize, key_num=key_num,
        ls_params=ls_params, detector_params=detector_params,
        source_type=source_type,
        loss=loss, thermal_params=thermal_params,
        loss_parameters=loss_parameters,
        phase_noise_coefficient=pnc,
    )
    if extra_kwargs:
        kwargs.update(extra_kwargs)
    return kwargs


# ═══════════════════════════════════════════════════════════════════════
#  Generic task builder (used by every sweep type)
# ═══════════════════════════════════════════════════════════════════════
def _build_tasks(points, point_to_spec, *, runtime, channel_parameters,
                 ls_lookup, det_lookup, base_loss_p, base_thermal_p,
                 base_keysize, key_num, protocols, extra_kwargs=None):
    """Build the (point x protocol) task list from a generic point->spec.

    ``point_to_spec(point)`` must return a dict with:
        'sweep_var', 'sweep_val'     (mandatory -- label the output axis)
        'distance', 'keysize'        (optional -- override the defaults)
        'loss_parameters'            (optional -- override base_loss_p)
        'thermal_params'             (optional -- override base_thermal_p)
        'ls_overrides'               (optional -- overlay on protocol ls_params)
        'kwarg_override'             (optional -- overlay on final kwargs)
    """
    fixed_dist, att, pfid_initial = channel_parameters
    tasks = []
    for point in points:
        spec = point_to_spec(point)
        sweep_var = spec["sweep_var"]
        sweep_val = spec["sweep_val"]
        distance  = spec.get("distance", fixed_dist)
        keysize   = spec.get("keysize",  base_keysize)
        lp_pt     = spec.get("loss_parameters", base_loss_p)
        tp_pt     = spec.get("thermal_params",  base_thermal_p)
        ls_over   = spec.get("ls_overrides")
        kwarg_o   = spec.get("kwarg_override", {})

        for proto in protocols:
            cfg = PROTOCOL_REGISTRY[proto]
            ls_p  = {**ls_lookup[cfg["ls_key"]], **(ls_over or {})}
            det_p = det_lookup[cfg["det_key"]]

            kwargs = _build_task_kwargs(
                distance=distance, keysize=keysize,
                ls_params=ls_p, detector_params=det_p,
                loss_parameters=lp_pt, thermal_params=tp_pt,
                runtime=runtime, attenuation=att,
                pfid_initial=pfid_initial, key_num=key_num,
                source_type=cfg["source_type"],
                extra_kwargs={**(extra_kwargs or {}), **kwarg_o},
            )
            tasks.append({"protocol": proto, "sweep_var": sweep_var,
                          "sweep_val": sweep_val, "kwargs": kwargs})
    return tasks


# ═══════════════════════════════════════════════════════════════════════
#  Worker + result wiring (used by every sweep type)
# ═══════════════════════════════════════════════════════════════════════
def _worker(task):
    """Run one simulation in a worker process."""
    proto = task["protocol"]
    sweep_var = task["sweep_var"]
    sweep_val = task["sweep_val"]
    res = run_qkd_simulation(proto, **task["kwargs"])
    out = {"protocol": proto, sweep_var: sweep_val,
           "skr": res["skr"], "qber": _safe_mean(res["qber"]),
           "throughputs": res["throughputs"], "latency": res["latency"],
           "loss": res["loss"], "rs": res["rs"]}
    if "visibility" in res:
        out["visibility"] = _safe_mean(res["visibility"])
    return out


def _fallback_result(proto, sweep_var, sweep_val):
    """NaN-filled result for a failed task; preserves CSV column shape."""
    fb = {"protocol": proto, sweep_var: sweep_val,
          "skr": np.nan, "qber": np.nan, "throughputs": np.nan,
          "latency": np.nan, "loss": np.nan, "rs": np.nan}
    if PROTOCOL_REGISTRY[proto]["needs_visibility"]:
        fb["visibility"] = np.nan
    return fb


def _collect_results(sweep_var, sweep_values, results_list, protocols):
    """Reorganise raw per-task results into wide-format columns."""
    data = {p: {} for p in protocols}
    for r in results_list:
        if r["protocol"] in data:
            data[r["protocol"]][r[sweep_var]] = r

    metrics = {sweep_var: np.array(sweep_values)}
    for proto in protocols:
        cols = list(_METRIC_COLS)
        if PROTOCOL_REGISTRY[proto]["needs_visibility"]:
            cols.append(_VIS_COL)
        for label, key in cols:
            metrics[f"{label}-{proto}"] = np.array(
                [data[proto].get(v, {}).get(key, np.nan) for v in sweep_values])
    return metrics


def _run_tasks(tasks, label, sweep_values, protocols, output_csv, max_workers):
    """Execute the task list in parallel, build the wide-format metrics
    table and save it as CSV. Returns the metrics dict."""
    if max_workers is None:
        max_workers = os.cpu_count() or 4

    total = len(tasks)
    print(f"[parallel] Launching {total} tasks across {max_workers} workers "
          f"({len(sweep_values)} {label}s x {len(protocols)} protocols)")

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_worker, t): t for t in tasks}
        for i, future in enumerate(as_completed(futures), 1):
            t = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                proto, val = t["protocol"], t["sweep_val"]
                print(f"\n[parallel] WARNING: {proto} @ {label}={val} "
                      f"failed: {exc}")
                results.append(_fallback_result(proto, label, val))
            print(f"\r[parallel] {i}/{total} done ({i/total*100:.1f}%)",
                  end="", flush=True)
    print()

    metrics = _collect_results(label, sweep_values, results, protocols)
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    pd.DataFrame(metrics).to_csv(output_csv, index=False)
    print(f"[parallel] Saved {output_csv}")
    return metrics


# ═══════════════════════════════════════════════════════════════════════
#  Public sweep APIs (thin wrappers -- define how points map to specs)
# ═══════════════════════════════════════════════════════════════════════
def sim_variable(sweep_var, sweep_values, *, runtime, channel_parameters,
                 ls_params, ls_params_cow, detector_params, detector_params_cow,
                 key_num, loss_parameters, thermal_params,
                 keysize=10000, protocols=None, output_csv=None,
                 max_workers=None, extra_kwargs=None):
    """Parallel sweep over a single kwarg of run_qkd_simulation.

    Args:
        sweep_var: name of the parameter to sweep. Must be a kwarg of
            run_qkd_simulation, e.g. 'distance', 'keysize', 'attenuation',
            'polarization_fidelity', 'eve_intercept_rate', ...
        sweep_values: iterable of values to assign to `sweep_var`.
        protocols: subset of PROTOCOL_REGISTRY keys (default: all).
        output_csv: where to save results. Default:
            data/metrics_variable-<sweep_var>.csv
        extra_kwargs: dict forwarded to every run_qkd_simulation call.

    Returns:
        dict of column_name -> np.ndarray (also saved as CSV).
    """
    if protocols is None:
        protocols = list(PROTOCOL_REGISTRY.keys())
    if output_csv is None:
        output_csv = f"data/metrics_variable-{sweep_var}.csv"
    sweep_values = list(sweep_values)

    ls_lookup  = {"ls_params": ls_params, "ls_params_cow": ls_params_cow}
    det_lookup = {"detector_params": detector_params,
                  "detector_params_cow": detector_params_cow}

    def point_to_spec(val):
        spec = {"sweep_var": sweep_var, "sweep_val": val}
        if sweep_var == "distance":
            spec["distance"] = val
        elif sweep_var == "keysize":
            spec["keysize"] = val
        else:
            # Any other run_qkd_simulation kwarg -- forwarded by name.
            spec["kwarg_override"] = {sweep_var: val}
        return spec

    tasks = _build_tasks(
        sweep_values, point_to_spec,
        runtime=runtime, channel_parameters=channel_parameters,
        ls_lookup=ls_lookup, det_lookup=det_lookup,
        base_loss_p=loss_parameters, base_thermal_p=thermal_params,
        base_keysize=keysize, key_num=key_num,
        protocols=protocols, extra_kwargs=extra_kwargs,
    )
    return _run_tasks(tasks, sweep_var, sweep_values, protocols,
                      output_csv, max_workers)


def sim_scenario(label, scenario_points, *, runtime, channel_parameters,
                 ls_params, ls_params_cow, detector_params, detector_params_cow,
                 keysize, key_num, base_loss_parameters, base_thermal_params,
                 materialize_fn, protocols=None, output_csv=None,
                 max_workers=None, extra_kwargs=None):
    """Parallel sweep where each point materialises a full parameter set.

    Args:
        label: name of the independent variable for the output CSV
            (e.g. 'hour').
        scenario_points: list of opaque tokens passed to materialize_fn.
        materialize_fn: callable
            (token, *, base_loss_parameters, base_thermal_params, ls_params)
            -> (loss_parameters, thermal_params, ls_overrides).
            ``ls_overrides`` can be None when the light-source params are
            unchanged by the scenario.
        ... (other args as in sim_variable)

    Returns:
        dict of column_name -> np.ndarray (also saved as CSV).
    """
    if protocols is None:
        protocols = list(PROTOCOL_REGISTRY.keys())
    if output_csv is None:
        output_csv = f"data/metrics_scenario-{label}.csv"
    scenario_points = list(scenario_points)

    ls_lookup  = {"ls_params": ls_params, "ls_params_cow": ls_params_cow}
    det_lookup = {"detector_params": detector_params,
                  "detector_params_cow": detector_params_cow}

    def point_to_spec(token):
        lp, tp, ls_overrides = materialize_fn(
            token,
            base_loss_parameters=base_loss_parameters,
            base_thermal_params=base_thermal_params,
            ls_params=ls_params,
        )
        return {"sweep_var": label, "sweep_val": token,
                "loss_parameters": lp, "thermal_params": tp,
                "ls_overrides": ls_overrides}

    tasks = _build_tasks(
        scenario_points, point_to_spec,
        runtime=runtime, channel_parameters=channel_parameters,
        ls_lookup=ls_lookup, det_lookup=det_lookup,
        base_loss_p=base_loss_parameters, base_thermal_p=base_thermal_params,
        base_keysize=keysize, key_num=key_num,
        protocols=protocols, extra_kwargs=extra_kwargs,
    )
    return _run_tasks(tasks, label, scenario_points, protocols,
                      output_csv, max_workers)


# ═══════════════════════════════════════════════════════════════════════
#  Execution driver
# ═══════════════════════════════════════════════════════════════════════
def run_simulation():
    start = time.time()

    # -- Light-source parameters
    # Frequency and wavelength from
    # 'THORLABS. DBR78TK, DBR79TK Low-Noise Laser Systems: user guide.
    #  Rev. B.: Thorlabs, Inc., 2025. Documento DOC-102639.'
    wavelength = 780          # nm
    frequency  = 8e6          # Hz
    ls_params     = {"frequency": frequency, "wavelength": wavelength, "mean_photon_num": 1}
    # mean_photon_num for COW from
    # 'STUCKI, D. et al. Fast and simple one-way quantum key distribution.
    #  Applied Physics Letters, v. 87, n. 19, 2 nov. 2005.'
    ls_params_cow = {"frequency": frequency, "wavelength": wavelength, "mean_photon_num": 0.5}

    # -- Detector parameters
    # Detector efficiency, dark counts, temporal resolution, and count rate
    # from 'THORLABS. SPDMH2/3/2F/3F: operation manual.
    #  Versão 1.2. Thorlabs GmbH, 2023. Documento MTN028160-D02.'
    count_rate      = 20e6    # Hz
    time_resolution = 1000    # ps
    det_template = {"efficiency": 0.65, "dark_count": 100,
                    "time_resolution": time_resolution,
                    "count_rate": count_rate}
    detector_params     = [dict(det_template) for _ in range(2)]
    detector_params_cow = [dict(det_template) for _ in range(3)]

    # -- Atmospheric / site parameters (base values; per-hour overrides
    #    come from materialize() in scenarios.py)
    temperature       = 298.15   # Kelvin
    friction_velocity = 200      # cm/s
    height_link       = 800      # cm
    link_altitude_m   = 720 + height_link / 100   # base topo + link mid-altitude
    sunrise = 6.516667
    sunset  = 17.566667          # Sunset/sunrise on May 11, 2026

    cn = cn2(time=12, sunset=sunset, sunrise=sunrise,
             temperature=temperature, wind_speed=friction_velocity / 100,
             rms_wind_speed=21, relative_humidity=0.47,
             height=link_altitude_m)

    # -- (distance_m, attenuation_dB/m, polarization_fidelity)
    channel_parameters = (700, 0.0002, None)

    # -- FSO loss parameters (forwarded as **kwargs to channel_FSO_loss)
    loss_parameters = {
        # Source of information on the factors that influence signal loss:
        "v_range":            10,                # Measured using 'WORLD METEOROLOGICAL ORGANIZATION. Guide to Instruments and Methods of Observation. 2024. p. 352–374'
        "receiver_radius":    10.3,              # sharpstar-optics.com/Products_1/79.html
        "pressure":           927,               # labmicro.iag.usp.br/Data/data_PMIAG.html
        "temperature":        temperature,       # labmicro.iag.usp.br/Data/data_PMIAG.html
        "w_0":                5,                 # Adopted as a first approximation until more precise measurements are available.
        "C_n2":               cn,                # computed above
        "R_0":                math.inf,          # For collimated beams, R_0 = math.inf is adopted
        "friction_velocity":  friction_velocity, # labmicro.iag.usp.br/Data/data_PMIAG.html
        "wind_speed_perp":    wind_speed_perp(link_altitude_m, friction_velocity / 100),
        "height":             height_link,       # transmitter/receiver mid-height
        "size_raindrop":      0.1,               # FADHIL, H. A. et al. Optimization of free space optics parameters: An optimum solution for bad weather conditions. v. 124, n. 19, p. 3969–3973, 1 out. 2013.
        "viscosity":          None,              # Sutherland, W. (1893), "The viscosity of gases and molecular force", Philosophical Magazine, S. 5, 36, pp. 507-531 (1893)
        "precipitation_rate": 0,                 # labmicro.iag.usp.br/Data/data_PMIAG.html
        "Q_scat":             2,                 # look *
    }
    # *Calculated using 'Prahl, S. (2026). miepython: Pure python calculation of Mie scattering (Version 3.2.0)
    #  [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.7949263' and 'Xiaohong Quan and Edward S. Fry, 
    #  "Empirical equation for the index of refraction of seawater," Appl. Opt. 34, 3477-3480 (1995)'


    # -- Thermal-noise parameters
    diameter_sensor = 1e-4         # m (https://media.thorlabs.com/globalassets/items/s/sp/spd/spdmh2/mtn028160-d02.pdf?v=0116030233)
    focal_distance  = 0.7004       # m (https://www.sharpstar-optics.com/Products_1/79.html)
    bandwidth       = 492767915845 # Hz  (~492 GHz, delta_lambda ~ 1 nm)
    thermal_params = {
        "delta_lambda_nm": (bandwidth * wavelength ** 2) / (_c * 1e9),
        "delta_t_ns":      1e9 / count_rate,
        "omega_fov_sr":    2 * math.pi * (1 - math.cos(
                              2 * math.atan(diameter_sensor / (2 * focal_distance)))),
        "a_R_cm":          loss_parameters["receiver_radius"],
        "B_sky":           1e-2, # look **
    }
    # **First approximation adopted of 'PIRANDOLA, S. Limits and security of free-space quantum 
    #   communications. Physical Review Research, v. 3, n. 1, 25 mar. 2021'
    #   A study of the natural source of brightness of the sky in ground-to-ground links is necessary.

    # -- Common kwargs reused by every variable-sweep
    common = dict(
        runtime=1000,
        channel_parameters=channel_parameters,
        ls_params=ls_params, ls_params_cow=ls_params_cow,
        detector_params=detector_params,
        detector_params_cow=detector_params_cow,
        key_num=1,
        loss_parameters=loss_parameters,
        thermal_params=thermal_params,
    )

    # -- Sweep #1: distance
    #sim_variable("distance", range(1000, 100001, 1000),
    #             keysize=10000, **common)

    # -- Sweep #2: keysize
    #sim_variable("keysize",
    #             [20, 45, 50, 100, 200, 400, 800, 1600,
    #              5000, 20000, 40000, 80000, 100000],
    #             **common)

    # -- Adding a new sweep is a one-liner. Examples:
    # sim_variable("attenuation", [1e-4, 2e-4, 4e-4, 8e-4],
    #              keysize=10_000, **common)
    # sim_variable("eve_intercept_rate", [0.1, 0.3, 0.5, 0.7, 0.9],
    #              keysize=10_000,
    #              protocols=["BB84+Eve", "B92+Eve", "COW+Eve"], **common)

    # -- Sweep #3: scenario by hour of day.
    # Bind site-specific parameters (sunrise/sunset/altitude) once via
    # functools.partial, leaving only the (token, base_loss_parameters,
    # base_thermal_params, ls_params) signature that sim_scenario expects.
    mat_fn = partial(materialize, sunrise=sunrise, sunset=sunset,
                     link_altitude_m=link_altitude_m)

    sim_scenario(
        label="hour",
        scenario_points=[0, 3, 6, 9, 12, 15, 18, 21],
        runtime=1000,
        channel_parameters=channel_parameters,
        ls_params=ls_params, ls_params_cow=ls_params_cow,
        detector_params=detector_params,
        detector_params_cow=detector_params_cow,
        keysize=10_000, key_num=1,
        base_loss_parameters=loss_parameters,
        base_thermal_params=thermal_params,
        materialize_fn=mat_fn,
    )

    pd.DataFrame({"Total_execution_time_(seconds)":
                  [time.time() - start]}
                 ).to_csv("data/simulator_metrics.csv", index=False)


if __name__ == "__main__":
    run_simulation()
