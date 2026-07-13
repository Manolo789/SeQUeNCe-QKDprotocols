import math
from datetime import datetime, timezone, timedelta
import time
import os
import warnings
from concurrent.futures import (ProcessPoolExecutor, as_completed, wait, FIRST_COMPLETED)

from functools import partial

import numpy as np
import pandas as pd

from sequence.components.optical_channel import QuantumChannel, ClassicalChannel, EveQuantumChannel
from sequence.components.thermal_noise_source import ThermalNoiseSource, _c
from sequence.kernel.event import Event
from sequence.kernel.process import Process
from sequence.kernel.timeline import Timeline
from sequence.qkd.BB84 import pair_bb84_protocols
from sequence.qkd.B92 import pair_b92_protocols
from sequence.qkd.COW import pair_cow_protocols
from sequence.topology.node import QKDNode, EveNode
from sequence.utils.encoding_cow import time_bin_cow
import sequence.utils.log as log

from QCLoss.loss import (
    f_velocity, outer_scale, inner_scale, viscosity_sutherland,
    channel_FSO_loss, cn2_horizontal_link, wind_speed_perp,
    make_atmospheric_phase_process)
from QCLoss.sky_radiance import (
    b_sky_at, n_background, detection_gate_from_detector)

from scenarios import diurnal_profile


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
    n_B = n_background(
        wavelength=ls_params["wavelength"],
        filter_bandwidth=thermal_params["filter_bandwidth"],
        detection_gate=thermal_params["detection_gate"],
        fov_solid_angle=thermal_params["fov_solid_angle"],
        receiver_radius=thermal_params["receiver_radius"],
        B_sky_si=thermal_params["B_sky"]
    )
    
    encoding = detector.owner.encoding if hasattr(detector.owner, "encoding") else None
    #src = ThermalNoiseSource(name=f"thermal_{detector.name}", timeline=tl, n_B=n_B, frequency=ls_params["frequency"], encoding_type=encoding)
    src = ThermalNoiseSource(name=f"thermal_{detector.name}", timeline=tl, n_B=n_B, frequency=ls_params["frequency"], encoding_type=encoding, detection_gate=thermal_params["detection_gate"])
    src.init()
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


def _configure_bob_cow_interferometer(bob, interferometer_phase_error, ls_frequency=None):
    """Set the Michelson phase error AND path difference on Bob's QSDetectorCOW.

    CORREÇÃO: o path_diff é calculado no construtor do QKDNode com a
    frequência DEFAULT da fonte, antes de update_lightsource_params. Aqui
    ele é ressincronizado com a frequência efetiva da fonte de Alice; sem
    isso, mudar ls_params_cow['frequency'] dessincroniza fonte e
    interferômetro (zero eventos de interferencia -> V=NaN -> SKR=0).
    """
    from sequence.components.qsdetector_cow import QSDetectorCOW
    from sequence.utils.encoding_cow import slot_period_ps
    for comp in bob.components.values():
        if isinstance(comp, QSDetectorCOW):
            comp.interferometer.phase_error = interferometer_phase_error
            if ls_frequency is not None:
                comp.interferometer.path_difference = slot_period_ps(ls_frequency)
            return


# ═══════════════════════════════════════════════════════════════════════
#  Unified simulation runner
# ═══════════════════════════════════════════════════════════════════════
def run_qkd_simulation(
    protocol, ls_params, detector_params, *,
    runtime=20, log_filename=-1, distance=1e3,
    polarization_fidelity=1, attenuation=2e-4,
    keysize=256, key_num=math.inf,
    source_type=None, loss=None, thermal_params=None,
    phase_noise_coefficient=0, interferometer_phase_error=0.20,
    eve_intercept_rate=0.9, eve_position=0.5, loss_parameters=None,
):
    """Run any QKD protocol registered in PROTOCOL_REGISTRY.

    Returns a dict with keys: qber, throughputs, latency, skr, loss, rs,
    plus 'visibility' for COW protocols.
    
    Observation 1: phase_noise_coefficient: laser phase noise (Wiener, rad/√m). Atmospheric turbulence enters 
            exclusively via atmospheric_phase_process (constructed from loss_parameters). Never populate 
            this parameter with phase_noise() from QCLoss/loss.py when loss_parameters is present—doing so double-counts the turbulence.
    Observation 2: If loss != None, then the 'attenuation' quantity is not considered, as there is an attenuation model different from the one normally used.

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
        qc_kwargs_other = dict(qc_kwargs)
        del qc_kwargs_other["loss"]
        del qc_kwargs_other["polarization_fidelity"]
        
        d_seg1 = distance * eve_position
        d_seg2 = distance * (1.0 - eve_position)
        loss_seg1 = _compute_loss(d_seg1, ls_params, loss_parameters)
        loss_seg2 = _compute_loss(d_seg2, ls_params, loss_parameters)
        qc0 = EveQuantumChannel("qc0", tl, eve_node=eve, eve_position=eve_position,
            atmospheric_phase_process_seg2=atm_eb,
            phase_noise_coefficient_seg2=phase_noise_coefficient,
            loss_seg1=loss_seg1,
            loss_seg2=loss_seg2,
            pf_seg1=polarization_fidelity,
            pf_seg2=polarization_fidelity,
            **qc_kwargs_other,
        )
        # total_distance = seg1 + seg2
        # total_loss = 1 - (1 - loss_seg1)*(1 - loss_seg2)
        qc_kwargs["loss"] = 1 - ((1 - loss_seg1) * (1 - loss_seg2))
    else:
        qc0 = QuantumChannel("qc0", tl, **qc_kwargs)
    qc1 = QuantumChannel("qc1", tl, **qc_kwargs)


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
        _configure_bob_cow_interferometer(bob, interferometer_phase_error, ls_params["frequency"])

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

    # Turbulência preserva a polarização em modo único (erro <1% em 144 km,
    # Fedrizzi et al., Nat. Phys. 5, 389 (2009)). Fixado em 1.0.
    pfid = pfid_initial if pfid_initial is not None else 1.0

    kwargs = dict(
        runtime=runtime, distance=distance,
        polarization_fidelity=pfid, attenuation=attenuation,
        keysize=keysize, key_num=key_num,
        ls_params=ls_params, detector_params=detector_params,
        source_type=source_type,
        loss=loss, thermal_params=thermal_params,
        loss_parameters=loss_parameters,
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
            #det_p = det_lookup[cfg["det_key"]]
            det_p = spec.get("detector_params", det_lookup[cfg["det_key"]])

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
    t_start = time.time()
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_worker, t): t for t in tasks}
        pending, i = set(futures), 0
        while pending:
            # Heartbeat: acorda a cada 30 s mesmo sem tarefa concluída,
            # para o usuário distinguir "trabalhando" de "travado".
            done, pending = wait(pending, timeout=30,
                                 return_when=FIRST_COMPLETED)
            if not done:
                print(f"\r[parallel] {i}/{total} done; "
                      f"{min(max_workers, len(pending))} running; "
                      f"elapsed {time.time() - t_start:.0f}s",
                      end="", flush=True)
                continue
            for future in done:
                i += 1
                t = futures[future]
                try:
                    results.append(future.result())
                except Exception as exc:
                    proto, val = t["protocol"], t["sweep_val"]
                    print(f"\n[parallel] WARNING: {proto} @ {label}={val} "
                          f"failed: {exc}")
                    results.append(_fallback_result(proto, label, val))
                print(f"\r[parallel] {i}/{total} done ({i/total*100:.1f}%), "
                      f"elapsed {time.time() - t_start:.0f}s",
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

    #def point_to_spec(val):
    #    spec = {"sweep_var": sweep_var, "sweep_val": val}
    #    if sweep_var == "distance":
    #        spec["distance"] = val
    #    elif sweep_var == "keysize":
    #        spec["keysize"] = val
    #    else:
    #        # Any other run_qkd_simulation kwarg -- forwarded by name.
    #        spec["kwarg_override"] = {sweep_var: val}
    #    return spec
    
        
    def point_to_spec(val):

        spec = {
            "sweep_var": sweep_var,
            "sweep_val": val
        }

        # cópias para não alterar os parâmetros base
        lp = dict(loss_parameters)
        tp = dict(thermal_params)
        ls_overrides = {}
        det_override = None

        if sweep_var == "distance":
            spec["distance"] = val
        elif sweep_var == "keysize":
            spec["keysize"] = val
        elif sweep_var in [
            "attenuation",
            "polarization_fidelity",
            "eve_intercept_rate",
            "eve_position",
            "interferometer_phase_error",
            "phase_noise_coefficient",
        ]:
            spec["kwarg_override"] = {sweep_var: val}

        elif sweep_var == "frequency":
            ls_overrides["frequency"] = val
        elif sweep_var == "efficiency":
            det_override = []
            for d in detector_params:
                x = dict(d)
                x["efficiency"] = val
                det_override.append(x)
            spec["detector_params"] = det_override
        elif sweep_var == "dark_count":
            det_override = []
            for d in detector_params:
                x = dict(d)
                x["dark_count"] = val
                det_override.append(x)
            spec["detector_params"] = det_override
        elif sweep_var == "visibility":
            lp["visibility"] = val
        elif sweep_var == "C_n2":
            lp["C_n2"] = val
        elif sweep_var == "temperature":
            lp["temperature"] = val
        elif sweep_var == "pressure":
            lp["pressure"] = val
        elif sweep_var == "wind_speed_perp":
            lp["wind_speed_perp"] = val
        elif sweep_var == "height_ag":
            lp["height_ag"] = val
        elif sweep_var == "receiver_radius":
            lp["receiver_radius"] = val
            tp["receiver_radius"] = val
        elif sweep_var == "precipitation_rate":
            lp["precipitation_rate"] = val
        elif sweep_var == "filter_bandwidth":
            tp["filter_bandwidth"] = val
        elif sweep_var == "fov_solid_angle":
            tp["fov_solid_angle"] = val
        else:
            raise ValueError(f"Sweep '{sweep_var}' não suportado.")

        if ls_overrides:
            spec["ls_overrides"] = ls_overrides

        spec["loss_parameters"] = lp
        spec["thermal_params"] = tp

        if det_override is not None:
            spec["detector_params"] = det_override

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
                 diurnal_profile_fn, protocols=None, output_csv=None,
                 max_workers=None, extra_kwargs=None):
    """Parallel sweep where each point materialises a full parameter set.

    Args:
        label: name of the independent variable for the output CSV
            (e.g. 'hour').
        scenario_points: list of opaque tokens passed to diurnal_profile_fn.
        diurnal_profile_fn: callable
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
        lp, tp, ls_overrides = diurnal_profile_fn(
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
    wavelength = 780e-9       # m
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
    #    come from diurnal_profile() in scenarios.py)
    temperature   = 298.15            # K
    pressure      = 92700.0           # Pa   (927 mbar)
    wind_speed    = 3.2               # m/s  (320 cm/s)
    height_link   = 8.0               # m ACIMA DO SOLO (800 cm)
    site_altitude = 720.0             # m ASL — só p/ wind_speed_perp (jato)
    latitude      = math.radians(-23.5615)   # LARC/EPUSP
    longitude     = math.radians(-46.7311)
    sunrise = 6.7833
    sunset  = 19.8833         # Sunset/sunrise on Feb 04, 2015 (Crepúsculo tipo -0.833º)

    friction_velocity = f_velocity(wind_speed, T_classification=7, height_ag=height_link)      # m/s
    viscosity = viscosity_sutherland(temperature)

    cn = cn2_horizontal_link(height_link, hour=12.0, sunrise=sunrise,
                             sunset=sunset, temperature=temperature,
                             wind_speed=wind_speed, relative_humidity=47.0)

    # -- (distance_m, attenuation_dB/m, polarization_fidelity)
    channel_parameters = (700, 0.0002, 1)

    # -- FSO loss parameters (forwarded as **kwargs to channel_FSO_loss)
    loss_parameters = {
        "visibility":         10e3,             # m   (Measured using 'WORLD METEOROLOGICAL ORGANIZATION. Guide to Instruments and Methods of Observation. 2024. p. 352–374')
        "receiver_radius":    0.103,            # m   (sharpstar-optics.com/Products_1/79.html)
        "pressure":           pressure,         # Pa  (labmicro.iag.usp.br/Data/data_PMIAG.html)
        "temperature":        temperature,      # K (labmicro.iag.usp.br/Data/data_PMIAG.html)
        "w_0":                0.05,             # m   (Adopted as a first approximation until more precise measurements are available.)
        "C_n2":               cn,
        "R_0":                math.inf,         # (For collimated beams, R_0 = math.inf is adopted)
        "friction_velocity":  friction_velocity,# m/s
        "wind_speed_perp":    wind_speed_perp(site_altitude, wind_speed),
        "height_ag":          height_link,      # m ACIMA DO SOLO
        "precipitation_rate": 0.0,              # m/s (1 mm/h = 2,78e-7 m/s, labmicro.iag.usp.br/Data/data_PMIAG.html)
    }

    # -- Thermal-noise parameters
    diameter_sensor = 1e-4         # m (https://media.thorlabs.com/globalassets/items/s/sp/spd/spdmh2/mtn028160-d02.pdf?v=0116030233)
    focal_distance  = 0.7004       # m (https://www.sharpstar-optics.com/Products_1/79.html)
    thermal_params = {
        "filter_bandwidth": 1e-9,                                 # m  (Δλ = 1 nm)
        "detection_gate":   detection_gate_from_detector(time_resolution), # s
        "fov_solid_angle":  2*math.pi*(1 - math.cos(math.atan(diameter_sensor/(2*focal_distance)))), # sr
        "receiver_radius":  loss_parameters["receiver_radius"],   # m
        "B_sky":            b_sky_at(datetime(2015, 2, 4, 15, 0, tzinfo=timezone.utc), latitude, longitude, ls_params["wavelength"], pressure=pressure), # **
    }
    # **First approximation adopted of 'PIRANDOLA, S. Limits and security of free-space quantum 
    #   communications. Physical Review Research, v. 3, n. 1, 25 mar. 2021'
    #   A study of the natural source of brightness of the sky in ground-to-ground links is necessary.

    extra_kwargs = None   # (antes: {'polarization': 'H'}; parametro removido)

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
        extra_kwargs=extra_kwargs,
    )

    sim_variable("distance", range(1000, 100001, 1000),
                 keysize=10000, **common)

    sim_variable("keysize",
                 [20, 45, 50, 100, 200, 400, 800, 1600,
                  5000, 20000, 40000, 80000, 100000],
                 **common)

    sim_variable("eve_intercept_rate", [0.1, 0.3, 0.5, 0.7, 0.9], keysize=10_000, protocols=["BB84+Eve", "B92+Eve", "COW+Eve"], **common)
                 
    sim_variable("efficiency", np.linspace(0.1,0.9,15), keysize=10000, **common)
    sim_variable("dark_count", [10,30,100,300,1000,3000,10000], keysize=10000, **common)
    sim_variable("frequency", [1e6,2e6,5e6,8e6,10e6,20e6,50e6], keysize=10000, **common)
    sim_variable("visibility", [100,200,500,1000,2000,5000,10000,20000,50000], keysize=10000, **common)
    sim_variable("C_n2", [1e-17,3e-17,1e-16,3e-16,1e-15], keysize=10000, **common)
    sim_variable("temperature", [273,282,293,303,308,313], keysize=10000, **common)
    sim_variable("pressure", [80000,85000,90000,92700, 95000,100000], keysize=10000, **common)

    wind_values = [wind_speed_perp(site_altitude,v) for v in [0.1,2,5,10,15,20]]

    sim_variable("wind_speed_perp", wind_values, keysize=10000, **common)
    sim_variable("height_ag", [2,5,8,10,20,50], keysize=10000, **common)
    sim_variable("receiver_radius", [0.025,0.05,0.075,0.10,0.15], keysize=10000, **common)
    sim_variable("filter_bandwidth", [0.1e-9,0.2e-9,0.5e-9,1e-9,2e-9,5e-9,10e-9], keysize=10000, **common)
    sim_variable("fov_solid_angle", np.linspace(1e-9,1e-6,20), keysize=10000, **common)
    
    mmh = 2.7778e-7

    sim_variable("precipitation_rate", [0.1*mmh, 1*mmh, 5*mmh, 10*mmh, 20*mmh, 30*mmh], keysize=10000, **common)
    sim_variable("interferometer_phase_error", np.linspace(0,1.1,20), keysize=10000, protocols=["COW","COW+Eve"], **common)
    sim_variable("eve_position", np.linspace(0.1,0.9,9), keysize=10000, protocols=["BB84+Eve","B92+Eve","COW+Eve"], **common)

    # -- Sweep #n: scenario by hour of day.
    # Bind site-specific parameters (sunrise/sunset/altitude) once via
    # functools.partial, leaving only the (token, base_loss_parameters,
    # base_thermal_params, ls_params) signature that sim_scenario expects.
    df = pd.read_csv("sensores/estação-solar-usp_Tabela01.dat", sep=',', skiprows=4, header=None, decimal='.', low_memory=False)
    df[0] = pd.to_datetime(df[0], format="%Y-%m-%d %H:%M:%S")
    diurnp_fn = partial(diurnal_profile, sunrise=sunrise, sunset=sunset,
                     site_altitude=site_altitude, latitude=latitude, longitude=longitude,
                     local_tz=timezone(timedelta(hours=-2)), date="2015-02-04", dataframe=df)
    
    sim_scenario(
        label="hour",
        scenario_points=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
        runtime=1000,
        channel_parameters=channel_parameters,
        ls_params=ls_params, ls_params_cow=ls_params_cow,
        detector_params=detector_params,
        detector_params_cow=detector_params_cow,
        keysize=10_000, key_num=1,
        base_loss_parameters=loss_parameters,
        base_thermal_params=thermal_params,
        diurnal_profile_fn=diurnp_fn,
        extra_kwargs=extra_kwargs,
    )

    pd.DataFrame({"Total_execution_time_(seconds)":
                  [time.time() - start]}
                 ).to_csv("data/simulator_metrics.csv", index=False)

if __name__ == "__main__":
    run_simulation()
