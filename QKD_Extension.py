"""Driver of the QKD comparison campaigns (BB84 / B92 / COW / BBM92 / E91).

The module exposes three layers:

1. :func:`run_qkd_simulation` -- a single simulation of any protocol of
   :data:`PROTOCOL_REGISTRY`, returning one result dict per run;
2. :func:`sim_variable` / :func:`sim_scenario` -- parallel parameter sweeps
   that turn those runs into the wide-format CSVs consumed by ``plot.py``;
3. :func:`run_simulation` -- the campaign driver, which executes the sweeps
   for TWO link models so that they can be compared point by point:

   * the *realistic free-space (aerial) link*, whose loss comes from
     ``QCLoss.loss.channel_FSO_loss`` (fog/aerosols, turbulence, rain) and
     whose receivers see the sky background of ``QCLoss.sky_radiance``;
   * the *attenuation-only reference link* (:func:`run_reference_link_simulation`),
     which reproduces the historical setup of commit 11ad36f: no atmospheric
     model at all, loss given by the textbook formula
     ``L = 1 - 10**(-alpha*d/10)`` and no sky background. Its CSVs are
     written to a separate directory so that both campaigns coexist.

Reproducibility and error bars
------------------------------
Every stochastic component is seeded from ONE global seed
(:func:`resolve_global_seed`): each task derives its own independent
substreams through :func:`derive_seed`, so a campaign is bit-for-bit
reproducible while different tasks stay statistically independent. The
global seed is written to ``data/simulator_metrics.csv`` together with the
campaign metadata needed to replicate it.

Every sweep point is simulated ``n_replicas`` times (independent seeds,
hence independent atmospheric realisations), and each replica generates
``key_num`` keys. Sampled metrics are exported as a mean plus a standard
deviation and a standard error (the ``*_std`` / ``*_sem`` CSV columns used
as error bars by ``plot.py``). The dispersion is computed ACROSS replicas
rather than across pooled keys, because keys of one run are clustered by
the atmospheric realisation they share -- see :func:`replicate_statistics`.
"""

import math
from datetime import datetime, timezone, timedelta
import hashlib
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
from sequence.qkd.BBM92 import pair_bbm92_protocols
from sequence.qkd.E91 import pair_e91_protocols
from sequence.topology.node import (QKDNode, EveNode,
    EntanglementSourceNode, MeasuringNode)
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
#  Reproducibility: one global seed, deterministic per-task substreams
# ═══════════════════════════════════════════════════════════════════════
#: Seed used when the caller gives none and the environment variable below
#: is unset. Any integer works; it is recorded in simulator_metrics.csv.
DEFAULT_GLOBAL_SEED = 20260805

#: Environment variable that overrides DEFAULT_GLOBAL_SEED without editing
#: the code (e.g. `QKD_GLOBAL_SEED=12345 python3 QKD_Extension.py`).
GLOBAL_SEED_ENV_VAR = "QKD_GLOBAL_SEED"


def resolve_global_seed(explicit_seed=None):
    """Return the campaign-wide seed, in decreasing order of priority.

    The precedence is ``explicit_seed`` > ``$QKD_GLOBAL_SEED`` >
    :data:`DEFAULT_GLOBAL_SEED`. The returned value is the ONLY entropy
    source of a campaign: every RNG stream of every task is derived from it
    by :func:`derive_seed`, so storing it (see
    :func:`save_simulator_metrics`) is enough to replay the whole campaign.

    Args:
        explicit_seed (int | None): seed passed programmatically.

    Returns:
        int: the global seed actually in use.

    Raises:
        ValueError: if the environment variable is set to a non-integer.
    """
    if explicit_seed is not None:
        return int(explicit_seed)
    env_value = os.environ.get(GLOBAL_SEED_ENV_VAR)
    if env_value is None or env_value.strip() == "":
        return DEFAULT_GLOBAL_SEED
    try:
        return int(env_value)
    except ValueError as exc:
        raise ValueError(
            f"{GLOBAL_SEED_ENV_VAR}={env_value!r} is not an integer.") from exc


def derive_seed(*parts):
    """Derive a reproducible 63-bit substream seed from arbitrary labels.

    The digest of the ``"|"``-joined string representation of ``parts`` is
    used instead of :func:`hash`, whose salt is randomised per interpreter,
    and instead of a plain ``seed + offset``, which correlates neighbouring
    streams. Two tasks therefore get independent-looking streams while the
    mapping (global seed, protocol, sweep point, role) -> seed stays stable
    across machines, Python versions and runs.

    Args:
        *parts: any objects identifying the stream (global seed, protocol
            name, sweep variable, sweep value, role such as "alice").

    Returns:
        int: a non-negative seed accepted by ``numpy.random.default_rng``.
    """
    label = "|".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(label, digest_size=8).digest(),
                          "big") >> 1


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
    # ── Entanglement-based protocols (BBM92 / E91) ─────────────────────────
    # Different topology (untrusted EPS in the middle, TWO measuring nodes),
    # but registered here so that the WHOLE existing pipeline — _worker,
    # _build_tasks, _collect_results, _run_tasks, sim_variable, sim_scenario —
    # is reused unchanged. run_qkd_simulation() dispatches on `entanglement`.
    # qkdtype extends the QKDNode numbering: 3 = BBM92, 4 = E91.
    # source_type: "eps" (1 pair/round) or "eps_poisson" (k~Poisson(mu) with
    # mu = ls_params["mean_photon_num"], multi-pair accidentals, Kržič §2.1.4).
    "BBM92": dict(qkdtype=3, pair_fn=pair_bbm92_protocols, encoding=None,
        log_module="BBM92", has_eve=False, needs_visibility=False,
        needs_chsh=False, entanglement=True,
        ls_key="ls_params", det_key="detector_params", source_type="eps",
    ),
    "E91": dict(qkdtype=4, pair_fn=pair_e91_protocols, encoding=None,
        log_module="E91", has_eve=False, needs_visibility=False,
        needs_chsh=True, entanglement=True,
        ls_key="ls_params", det_key="detector_params", source_type="eps",
    ),
    "BBM92+Eve": dict(qkdtype=3, pair_fn=pair_bbm92_protocols, encoding=None,
        log_module="BBM92", has_eve=True, needs_visibility=False,
        needs_chsh=False, entanglement=True,
        ls_key="ls_params", det_key="detector_params", source_type="eps",
    ),
    "E91+Eve": dict(qkdtype=4, pair_fn=pair_e91_protocols, encoding=None,
        log_module="E91", has_eve=True, needs_visibility=False,
        needs_chsh=True, entanglement=True,
        ls_key="ls_params", det_key="detector_params", source_type="eps",
    ),
}

#: convenience protocol groups (sweep defaults use the prepare-and-measure
#: group, preserving the historical behaviour of sim_variable/sim_scenario).
PREPARE_MEASURE_PROTOCOLS = [p for p, c in PROTOCOL_REGISTRY.items()
                             if not c.get("entanglement")]
ENTANGLEMENT_PROTOCOLS = [p for p, c in PROTOCOL_REGISTRY.items()
                          if c.get("entanglement")]

# Metric column names used in the output CSVs (label, dict-key).
_METRIC_COLS = [
    ("R_sk", "skr"), ("QBER", "qber"), ("Throughputs", "throughputs"),
    ("Latency", "latency"), ("Loss", "loss"), ("R_s", "rs"),
]
_VIS_COL = ("Visibility", "visibility")
_CHSH_COL = ("CHSH_S", "chsh_S")   # E91: same conditional-column pattern as _VIS_COL

# Metrics with one independent sample per generated key: they are exported
# as mean + "_std" (sample standard deviation) + "_sem" (standard error).
_SAMPLED_METRIC_KEYS = {"skr", "qber", "throughputs", "rs",
                        _VIS_COL[1], _CHSH_COL[1]}
# Loss is deterministic for a given point and `latency` is measured once per
# run (first key only), so neither carries an error bar.

#: Number of keys that actually entered the statistics of a point.
_NKEYS_COL = ("N_keys", "n_keys")

#: Number of independent replicas that contributed to a point.
_NREPLICAS_COL = ("N_replicas", "n_replicas")

#: Output directory of the realistic free-space (aerial link) campaign.
DEFAULT_DATA_DIR = "data"
#: Output directory of the attenuation-only reference campaign, kept apart
#: so that both link models can be plotted and compared side by side.
REFERENCE_LINK_DATA_DIR = os.path.join("data", "reference_link")

#: Keys generated per replica. Consecutive keys advance the RNG stream, so
#: they ARE distinct realisations of the detection/eavesdropping noise --
#: but they share the run's atmospheric phase realisation, which is
#: pre-generated once per simulation. They therefore measure the
#: WITHIN-realisation (temporal) variation only.
KEY_NUM_FOR_STATISTICS = 5

#: Independent replicas of every sweep point. A replica is a full
#: re-simulation under a different derived seed, so it re-draws EVERYTHING,
#: including the atmospheric piston and Eve's attack pattern. This is what
#: turns the error bar into an ensemble quantity (see
#: :func:`replicate_statistics`); values below 2 fall back to the
#: within-run error bar. Cost per point scales as key_num * n_replicas.
N_REPLICAS_FOR_STATISTICS = 5

#: Sweeps that only make sense with an atmospheric model configured.
_ATMOSPHERIC_SWEEPS = frozenset({
    "atm_visibility", "C_n2", "temperature", "pressure", "height_ag",
    "ground_wind_speed", "wind_speed_perp", "receiver_radius",
    "precipitation_rate"})
#: Sweeps that only make sense with a sky-background model configured.
_THERMAL_SWEEPS = frozenset({"filter_bandwidth", "fov_solid_angle"})
#: Suffixes appended to a sampled metric label to build its CSV columns.
_STAT_SUFFIXES = ("", "_std", "_sem")

# Keys reserved by the _worker result dict. A sweep_var with one of these
# names would collide with a metric and blank out (NaN) the protocol columns
# of the CSV (historical case: sweep "visibility" [atmospheric] vs. metric
# "visibility" [COW interferometer]).
_RESERVED_RESULT_KEYS = (
    {"protocol", "global_seed", "replica", "samples",
     _NKEYS_COL[1], _NREPLICAS_COL[1]}
    | {k for _, k in _METRIC_COLS}
    | {_VIS_COL[1], _CHSH_COL[1]}
    | {f"{k}{s}" for k in _SAMPLED_METRIC_KEYS for s in _STAT_SUFFIXES})


# ═══════════════════════════════════════════════════════════════════════
#  Metric helpers
# ═══════════════════════════════════════════════════════════════════════
def binary_entropy(Q):
    """Binary Shannon entropy H2(Q) = -Q*log2(Q) - (1-Q)*log2(1-Q).

    ``Q`` is clipped to [0, 1] before the evaluation, so QBER estimates
    slightly outside the physical range (numerical noise) do not raise.

    Args:
        Q (float): error probability.

    Returns:
        float: H2(Q) in bits; 0 at the endpoints Q = 0 and Q = 1.
    """
    Q = max(0.0, min(1.0, Q))
    if Q == 0 or Q == 1:
        return 0
    return -Q * math.log2(Q) - (1 - Q) * math.log2(1 - Q)


def _safe_mean(lst, default=np.nan):
    """np.nanmean(lst) but tolerant of None / empty / scalar inputs.

    Args:
        lst: sequence of samples, a bare scalar, or None.
        default: value returned for an empty/None/all-NaN input.

    Returns:
        float: the mean of the finite entries, or ``default``.
    """
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


def sample_statistics(samples):
    """Mean, sample standard deviation and standard error of a flat sample.

    The unbiased (ddof=1) estimator is used, so the dispersion is
    undefined for a single value. This is the WITHIN-run estimator: it
    treats every entry as an independent draw and is applied to the keys
    of one simulation, which share that run's atmospheric realisation. Use
    :func:`replicate_statistics` for the ensemble error bar.

    Args:
        samples: sequence of per-key values (may contain NaN) or None.

    Returns:
        tuple: (mean, std, sem, n) where ``std``/``sem`` are NaN when fewer
        than two finite samples are available and ``n`` is the number of
        finite samples that entered the statistics.
    """
    if samples is None:
        return np.nan, np.nan, np.nan, 0
    values = np.asarray(samples, dtype=float).ravel()
    values = values[np.isfinite(values)]
    n = int(values.size)
    if n == 0:
        return np.nan, np.nan, np.nan, 0
    mean = float(values.mean())
    if n < 2:
        return mean, np.nan, np.nan, n
    std = float(values.std(ddof=1))
    return mean, std, std / math.sqrt(n), n

def replicate_statistics(per_replica_samples):
    """Ensemble statistics of a metric measured over independent replicas.

    Keys generated inside ONE simulation share the run's non-resampled
    state -- most importantly the atmospheric phase realisation, which is
    pre-generated once per run. They are therefore CLUSTERED samples:
    pooling them and dividing by sqrt(N) would understate the uncertainty,
    because consecutive keys explore time windows of a single turbulence
    realisation rather than the turbulence ensemble.

    With two or more replicas the estimator is computed at the replica
    level (the standard cluster-robust choice): each replica contributes
    its own mean, and the dispersion is taken ACROSS those means, so the
    error bar covers both the within-run noise and the run-to-run spread
    of the atmospheric/eavesdropping realisation. With a single replica it
    falls back to pooling the keys, which measures only the within-run
    (temporal) variation -- see :func:`sample_statistics`.

    Args:
        per_replica_samples: sequence of per-key sample lists, one list per
            replica (a replica being one full simulation with its own seed).

    Returns:
        tuple: (mean, std, sem, n_keys, n_replicas) where ``n_keys`` counts
        every finite key sample pooled over replicas and ``n_replicas``
        counts the replicas that produced at least one key.
    """
    groups = []
    for samples in (per_replica_samples or []):
        if samples is None:
            continue
        values = np.asarray(samples, dtype=float).ravel()
        values = values[np.isfinite(values)]
        if values.size:
            groups.append(values)

    n_keys = int(sum(g.size for g in groups))
    n_replicas = len(groups)
    if n_replicas == 0:
        return np.nan, np.nan, np.nan, 0, 0
    if n_replicas == 1:
        mean, std, sem, _ = sample_statistics(groups[0])
        return mean, std, sem, n_keys, 1

    # Replicas are weighted equally: each one is a draw of the ensemble,
    # regardless of how many keys it managed to produce.
    means = np.array([g.mean() for g in groups], dtype=float)
    mean = float(means.mean())
    std = float(means.std(ddof=1))
    return mean, std, std / math.sqrt(n_replicas), n_keys, n_replicas


def _empty_metrics(protocol):
    """Zero-filled metric bundle for a run that produced no key.

    Args:
        protocol: the protocol instance whose (empty) run is summarised.

    Returns:
        dict: the same shape as :func:`_collect_metrics`, with empty sample
        lists so that the statistics collapse to NaN instead of failing.
    """
    return dict(qber=list(protocol.error_rates), throughputs=0.0,
                latency=protocol.latency, skr=0.0, rs=0.0,
                skr_samples=[], rs_samples=[],
                throughput_samples=list(protocol.throughputs))


def _collect_metrics(protocol, f_ec=1.0):
    """Per-key metrics of a finished run: QBER, throughput, latency, SKR, R_s.

    The asymptotic secure fraction of Krzic Eq. (2.11) is applied to every
    key separately, which is what makes the per-point mean and error bar
    meaningful:

        R_s  = sifted bits / qubits sent            (per key)
        R_sk = R_s * [1 - f_EC*H2(E) - H2(E)]       (per key)

    The denominator (``send_bits_length``) is the number of qubits emitted
    per train, so R_s and R_sk are in bits per qubit sent for BOTH protocol
    families and are directly comparable.

    Args:
        protocol: protocol instance on Alice's side after ``Timeline.run``.
        f_ec (float): error-correction inefficiency factor (1.0 = Shannon).

    Returns:
        dict: means (``qber``, ``throughputs``, ``latency``, ``skr``, ``rs``)
        plus the per-key sample lists used for the error bars
        (``skr_samples``, ``rs_samples``, ``throughput_samples``); ``qber``
        is itself the per-key list.
    """
    qber_list = protocol.error_rates
    if not qber_list or protocol.send_bits_length == 0:
        return _empty_metrics(protocol)

    rs_list, skr_list = [], []
    for i, e in enumerate(qber_list):
        rs = protocol.sifted_bits_length[i] / protocol.send_bits_length
        skr_list.append(max(0.0, rs * (1 - f_ec * binary_entropy(e)
                                       - binary_entropy(e))))
        rs_list.append(rs)
    return dict(qber=qber_list, throughputs=_safe_mean(protocol.throughputs, 0.0),
                latency=protocol.latency,
                skr=float(np.mean(skr_list)), rs=float(np.mean(rs_list)),
                skr_samples=skr_list, rs_samples=rs_list,
                throughput_samples=list(protocol.throughputs))


def _collect_cow_metrics(protocol, visibility, ls_params, loss, f_ec=1.0):
    """COW metrics with visibility-adjusted SKR (DOI 10.1063/1.2126792).

    The COW security witness is the interference visibility V of the
    monitoring line, so Eve's information is bounded by
    ``r + (1 - V)*(1 + e^{-mu*t})/(2*e^{-mu*t})`` with ``r = mu*(1 - t)``
    the fraction of pulses vulnerable to photon-number splitting and
    ``t = 1 - loss`` the channel transmission.

    Args:
        protocol: COW protocol instance on Alice's side.
        visibility (list[float]): per-key monitoring-line visibility.
        ls_params (dict): light-source parameters (uses ``mean_photon_num``).
        loss (float): end-to-end channel loss fraction in [0, 1].
        f_ec (float): error-correction inefficiency factor.

    Returns:
        dict: same shape as :func:`_collect_metrics`.
    """
    qber_list = protocol.error_rates
    if not qber_list or protocol.send_bits_length == 0:
        return _empty_metrics(protocol)

    mu = ls_params["mean_photon_num"]
    t = 1 - loss
    r = mu * (1 - t)
    rs_list, skr_list = [], []
    for i, e in enumerate(qber_list):
        rs = protocol.sifted_bits_length[i] / protocol.send_bits_length
        v = visibility[i]
        eve_info = r + ((1 - v) * (1 + math.exp(-mu * t)) / (2 * math.exp(-mu * t)))
        skr_list.append(max(0.0, rs * (1 - f_ec * binary_entropy(e) - eve_info)))
        rs_list.append(rs)
    return dict(qber=qber_list, throughputs=_safe_mean(protocol.throughputs, 0.0),
                latency=protocol.latency,
                skr=float(np.mean(skr_list)), rs=float(np.mean(rs_list)),
                skr_samples=skr_list, rs_samples=rs_list,
                throughput_samples=list(protocol.throughputs))


def resolve_entanglement_arms(distance, charlie_position=0.5,
                              distance_ac=None, distance_cb=None):
    """Split the Alice--Bob separation into the two source arms.

    ``distance`` has the SAME meaning for every protocol in the registry:
    the total Alice--Bob separation. For the entanglement-based family the
    untrusted source (Charlie) sits somewhere between them, and

        distance = distance_ac + distance_cb

    with ``charlie_position`` in (0, 1) fixing where along the link he is:

        distance_ac = charlie_position * distance
        distance_cb = (1 - charlie_position) * distance

    so ``charlie_position = 0.5`` is the symmetric configuration (each arm
    carries half of the link, the case that extends the reachable range
    w.r.t. a prepare-and-measure link of the same total length) while
    ``charlie_position -> 0`` puts the source at Alice's site.

    When ``charlie_position`` is None the two arms are taken from the
    optional ``distance_ac`` / ``distance_cb`` parameters instead, and the
    total separation is their sum (asymmetric links whose split is given
    directly rather than as a fraction).

    Args:
        distance (float): total Alice--Bob separation [m]; may be None if
            both arms are given explicitly.
        charlie_position (float): fraction of ``distance`` on Alice's arm,
            or None to use the explicit arms.
        distance_ac (float): Alice--Charlie arm [m] (used when
            ``charlie_position`` is None).
        distance_cb (float): Charlie--Bob arm [m] (idem).

    Returns:
        tuple: (distance_ac, distance_cb, distance_total) in metres.
    """
    if distance is not None and charlie_position is not None:
        position = float(charlie_position)
        if not 0.0 < position < 1.0:
            raise ValueError(
                f"charlie_position must lie in (0, 1); got {position!r} "
                "(0 = source at Alice's site, 1 = source at Bob's site).")
        total = float(distance)
        return position * total, (1.0 - position) * total, total

    if distance_ac is None or distance_cb is None:
        raise ValueError(
            "with charlie_position=None (or distance=None) both distance_ac "
            "and distance_cb must be given explicitly; got "
            f"distance_ac={distance_ac!r}, distance_cb={distance_cb!r}.")
    arm_ac, arm_cb = float(distance_ac), float(distance_cb)
    if arm_ac < 0 or arm_cb < 0:
        raise ValueError("distance_ac and distance_cb must be non-negative.")
    return arm_ac, arm_cb, arm_ac + arm_cb


def _n_background_photons(ls_params, thermal_params):
    """Background photons per detection gate at one receiver.

    Single source of truth for the ``n_background`` call, shared by
    ``_attach_thermal_noise`` (prepare-and-measure detectors) and the
    entanglement runner (per-round noise probability).
    """
    return n_background(
        wavelength=ls_params["wavelength"],
        filter_bandwidth=thermal_params["filter_bandwidth"],
        detection_gate=thermal_params["detection_gate"],
        fov_solid_angle=thermal_params["fov_solid_angle"],
        receiver_radius=thermal_params["receiver_radius"],
        B_sky_si=thermal_params["B_sky"]
    )


def _attach_thermal_noise(tl, qsdetector, ls_params, thermal_params, seed=0):
    """Create and connect a ThermalNoiseSource to the receiver front end.

    The receiver registered here is the COMPLETE QSDetector, never
    ``detectors[0]``: ``QSDetector.get()`` owns the routing element
    (``BeamSplitter`` for polarisation, t_B splitter + Michelson for COW),
    so injecting the background photon into the detector array directly
    would (a) make the random polarisation drawn by
    ``ThermalNoiseSource._random_state`` dead code, (b) send every
    background count to detector 0 -- i.e. always the same bit, halving
    the sky contribution to the QBER and biasing the raw key towards 0 --
    and (c) keep the background away from DM1/DM2 in COW, leaving the
    monitoring-line visibility insensitive to the sky brightness.

    Args:
        tl (Timeline): timeline that owns the simulation entities.
        qsdetector (QSDetector): full receiver front end of Bob.
        ls_params (dict): light-source parameters (wavelength, frequency).
        thermal_params (dict): sky-background parameters (see
            :func:`_n_background_photons`).
        seed (int): seed of the background source RNG.

    Returns:
        ThermalNoiseSource: the source, already initialised and registered.
    """
    n_B = _n_background_photons(ls_params, thermal_params)

    owner = getattr(qsdetector, "owner", None)
    encoding = owner.encoding if hasattr(owner, "encoding") else None
    src = ThermalNoiseSource(name=f"thermal_{qsdetector.name}", timeline=tl,
                             n_B=n_B, frequency=ls_params["frequency"],
                             encoding_type=encoding,
                             detection_gate=thermal_params["detection_gate"],
                             seed=seed)
    src.init()
    src.add_receiver(qsdetector)
    tl.entities[src.name] = src
    return src


# ═══════════════════════════════════════════════════════════════════════
#  Building blocks for the simulation runner
# ═══════════════════════════════════════════════════════════════════════
def _setup_atm_processes(is_cow, has_eve, loss_parameters, ls_params, distance,
                         eve_position, stop_time_ps, seed=0):
    """Pre-generate atmospheric piston processes for one simulation run.

    Only the COW protocol is phase sensitive, and the process needs the
    atmospheric description, so the pair is ``(None, None)`` for any other
    protocol or whenever ``loss_parameters`` is absent (which is exactly
    the attenuation-only reference link).

    Args:
        is_cow (bool): whether the protocol reads the channel phase.
        has_eve (bool): whether the link is split by an eavesdropper.
        loss_parameters (dict | None): atmospheric/FSO parameters.
        ls_params (dict): light-source parameters (uses ``wavelength``).
        distance (float): total Alice--Bob separation [m].
        eve_position (float): fraction of the link where Eve sits.
        stop_time_ps (float): timeline horizon to pre-generate [ps].
        seed (int): run seed; each segment derives its own substream.

    Returns:
        tuple: (atm_ab, atm_eb) -- phase processes for the segments
        Alice -> (Bob or Eve) and Eve -> Bob; ``atm_eb`` is None without Eve.
    """
    if not is_cow or loss_parameters is None:
        return None, None

    if has_eve:
        d1 = distance * eve_position
        d2 = distance * (1.0 - eve_position)
        atm_ab = make_atmospheric_phase_process(distance=d1, timeline_stop_time_ps=stop_time_ps, ls_params=ls_params, loss_parameters=loss_parameters, seed=derive_seed(seed, "atm_alice_eve"))
        atm_eb = make_atmospheric_phase_process(distance=d2, timeline_stop_time_ps=stop_time_ps, ls_params=ls_params, loss_parameters=loss_parameters, seed=derive_seed(seed, "atm_eve_bob"))
        return atm_ab, atm_eb

    atm_ab = make_atmospheric_phase_process(distance=distance, timeline_stop_time_ps=stop_time_ps, ls_params=ls_params, loss_parameters=loss_parameters, seed=derive_seed(seed, "atm_alice_bob"))
    return atm_ab, None


def _configure_bob_cow_interferometer(bob, interferometer_phase_error, ls_frequency=None):
    """Set the Michelson phase error AND path difference on Bob's QSDetectorCOW.

    ``path_diff`` is computed in the QKDNode constructor from the DEFAULT
    source frequency, i.e. before ``update_lightsource_params`` runs, so it
    is resynchronised here with the effective frequency of Alice's source.
    Without this, changing ``ls_params_cow['frequency']`` desynchronises
    source and interferometer (no interference event at all -> V = NaN ->
    SKR = 0).

    Args:
        bob (QKDNode): receiving node hosting the QSDetectorCOW.
        interferometer_phase_error (float): phase error of the Michelson.
        ls_frequency (float | None): effective source frequency [Hz]; when
            None the constructor value of the path difference is kept.
    """
    from sequence.components.qsdetector_cow import QSDetectorCOW
    from sequence.utils.encoding_cow import slot_period_ps
    for comp in bob.components.values():
        if isinstance(comp, QSDetectorCOW):
            comp.interferometer.phase_error = interferometer_phase_error
            if ls_frequency is not None:
                comp.interferometer.path_difference = slot_period_ps(ls_frequency)
            return


def attenuation_loss(distance, attenuation):
    """Textbook link loss ``L = 1 - 10**(-alpha*d/10)`` of a lossy channel.

    This is the loss model of the attenuation-only reference link (the
    historical behaviour of commit 11ad36f): a single dB/m coefficient, no
    atmospheric physics. It is also the value reported in the ``Loss``
    column whenever no explicit ``loss`` override is supplied, so that the
    two campaigns always document the transmission they actually used.

    Args:
        distance (float): link length [m].
        attenuation (float): attenuation coefficient [dB/m].

    Returns:
        float: loss fraction in [0, 1], or NaN if the inputs are missing.
    """
    if distance is None or attenuation is None:
        return np.nan
    return 1.0 - 10.0 ** ((distance * attenuation) / -10.0)


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
    charlie_position=0.5, distance_ac=None, distance_cb=None,
    f_ec=1.0, bell_state="psi_minus", seed=0,
    classical_delay_offset_ps=0.0, enforce_bell_violation=True,
):
    """Run any QKD protocol registered in PROTOCOL_REGISTRY.

    Returns a dict with keys: qber, throughputs, latency, skr, loss, rs,
    the per-key sample lists that feed the error bars (skr_samples,
    rs_samples, throughput_samples), plus 'visibility' for COW protocols
    and 'chsh_S' for E91 protocols.

    ``key_num`` keys are generated per call and each one is an independent
    Monte Carlo realisation, so raising it is what turns a point into a
    mean with a dispersion (see :func:`sample_statistics`). ``seed`` is the
    run seed: every RNG of the run (Alice, Bob, Eve, the atmospheric piston
    and the sky background) is a substream derived from it via
    :func:`derive_seed`, so the run is reproducible and independent from
    its neighbours.

    Observation 1: phase_noise_coefficient: laser phase noise (Wiener, rad/√m). Atmospheric turbulence enters 
            exclusively via atmospheric_phase_process (constructed from loss_parameters). Never populate 
            this parameter with phase_noise() from QCLoss/loss.py when loss_parameters is present—doing so double-counts the turbulence.
    Observation 2: If loss != None, then the 'attenuation' quantity is not considered, as there is an attenuation
            model different from the one normally used. With loss=None AND loss_parameters=None the channel falls
            back to the plain dB/m formula (see :func:`attenuation_loss`), which is the attenuation-only
            reference link used as a baseline for the realistic free-space campaign.
    Observation 3: For the entanglement-based protocols (BBM92/E91) the shared
            contracts are reused — ls_params (frequency, wavelength,
            mean_photon_num for "eps_poisson" sources), detector_params[0]
            (efficiency, dark_count, time_resolution), loss/loss_parameters,
            thermal_params, eve_intercept_rate/eve_position — AND, since the
            entanglement family is keysize-oriented too, runtime/keysize/
            key_num as well: `push(keysize, key_num, run_time)` drives both
            families, the sifted bits accumulate over successive trains and
            a key is emitted whenever `len(key_bits) >= keysize`. Only the
            phase-noise/interferometer kwargs remain prepare-and-measure
            specific; charlie_position/distance_ac/distance_cb/bell_state/
            seed apply ONLY to the entanglement family.
            `f_ec` applies to EVERY protocol (same 1 - f_EC*H2(E) - H2(E)
            key fraction), so the `f_ec` sweep is meaningful for the whole
            registry and the families stay comparable.
    Observation 4: `distance` is ALWAYS the total Alice--Bob separation. For the
            entanglement family the untrusted source sits between them and
            the arms follow from `charlie_position`
            (distance = distance_ac + distance_cb, see
            :func:`resolve_entanglement_arms`); with charlie_position=None
            the arms are read from `distance_ac`/`distance_cb` instead. This
            makes a `distance` sweep mean the same thing for every protocol.
    """
    cfg = PROTOCOL_REGISTRY[protocol]

    # Entanglement-based protocols: same registry, same sweep pipeline,
    # dedicated topology (EPS in the middle + two measuring nodes).
    if cfg.get("entanglement"):
        return _run_entanglement_qkd(
            cfg, ls_params, detector_params,
            distance=distance, charlie_position=charlie_position,
            distance_ac=distance_ac, distance_cb=distance_cb,
            attenuation=attenuation, loss=loss,
            loss_parameters=loss_parameters, thermal_params=thermal_params,
            polarization_fidelity=polarization_fidelity,
            source_type=source_type if source_type is not None else cfg["source_type"],
            eve_intercept_rate=eve_intercept_rate, eve_position=eve_position,
            runtime=runtime, keysize=keysize, key_num=key_num,
            f_ec=f_ec, bell_state=bell_state, seed=seed,
            classical_delay_offset_ps=classical_delay_offset_ps,
            enforce_bell_violation=enforce_bell_violation,
        )

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
    atm_ab, atm_eb = _setup_atm_processes(is_cow, cfg["has_eve"], loss_parameters, ls_params, distance, eve_position, tl.stop_time, seed=seed)

    # Quantum channels (qc0 carries Eve when applicable).
    qc_kwargs = dict(
        distance=distance,
        polarization_fidelity=polarization_fidelity,
        attenuation=attenuation, loss=loss,
        atmospheric_phase_process=atm_ab,
        phase_noise_coefficient=phase_noise_coefficient,
    )

    if cfg["has_eve"]:
        eve_kwargs = dict(intercept_rate=eve_intercept_rate,
                          seed=derive_seed(seed, "eve"))
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
        # Reference link (no atmospheric model): split the dB/m loss.
        if loss_seg1 is None or loss_seg2 is None:
            loss_seg1 = attenuation_loss(d_seg1, attenuation)
            loss_seg2 = attenuation_loss(d_seg2, attenuation)
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


    # Classical channels. The post-processing offset is explicit, shared
    # with _run_entanglement_qkd and null by default (see the signature).
    cc0 = ClassicalChannel("cc0", tl, distance=distance)
    cc1 = ClassicalChannel("cc1", tl, distance=distance)
    cc0.delay += classical_delay_offset_ps
    cc1.delay += classical_delay_offset_ps

    # Nodes.
    node_kwargs = dict(stack_size=1, qkdtype=cfg["qkdtype"], source_type=src_type)
    if cfg["encoding"] is not None:
        node_kwargs["encoding"] = cfg["encoding"]

    alice = QKDNode("alice", tl, **node_kwargs)
    alice.set_seed(derive_seed(seed, "alice"))
    for k, v in ls_params.items():
        alice.update_lightsource_params(k, v)

    bob = QKDNode("bob", tl, **node_kwargs)
    bob.set_seed(derive_seed(seed, "bob"))
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
        bob_qsd = bob.components[bob.first_component_name]
        _attach_thermal_noise(tl, bob_qsd, ls_params, thermal_params,
                              seed=derive_seed(seed, "thermal_bob"))
    tl.run()

    # Reference link (loss=None): report the dB/m loss the channel applied.
    effective_loss = loss if loss is not None else attenuation_loss(distance, attenuation)

    proto_obj = alice.protocol_stack[0]
    if cfg["needs_visibility"]:
        vis = proto_obj.visibility
        res = _collect_cow_metrics(proto_obj, vis, ls_params, effective_loss, f_ec)
        res["visibility"] = vis
    else:
        res = _collect_metrics(proto_obj, f_ec)
    res["loss"] = effective_loss
    return res


# ═══════════════════════════════════════════════════════════════════════
#  Backwards-compatible thin wrappers (preserve original tuple API)
#  Each simulation_<PROTOCOL>() forwards its arguments to
#  run_qkd_simulation() with the matching registry key and returns the
#  positional tuple documented in _unpack().
# ═══════════════════════════════════════════════════════════════════════
def _unpack(res):
    """Flatten a result dict into the historical positional tuple.

    Kept so that the ``simulation_*`` helpers below preserve the tuple API
    of the pre-registry code; the trailing elements only exist for the
    protocols that define them (COW visibility, E91 Bell parameter).

    Args:
        res (dict): output of :func:`run_qkd_simulation`.

    Returns:
        tuple: (qber, throughputs, latency, skr, loss, rs[, visibility]
        [, chsh_S]).
    """
    base = (res["qber"], res["throughputs"], res["latency"], res["skr"], res["loss"], res["rs"])
    if "visibility" in res:
        base += (res["visibility"],)
    if "chsh_S" in res:
        base += (res["chsh_S"],)
    return base


def simulation_BB84(*a, **kw):     return _unpack(run_qkd_simulation("BB84", *a, **kw))
def simulation_B92(*a, **kw):      return _unpack(run_qkd_simulation("B92", *a, **kw))
def simulation_COW(*a, **kw):      return _unpack(run_qkd_simulation("COW", *a, **kw))
def simulation_BB84_Eve(*a, **kw): return _unpack(run_qkd_simulation("BB84+Eve", *a, **kw))
def simulation_B92_Eve(*a, **kw):  return _unpack(run_qkd_simulation("B92+Eve", *a, **kw))
def simulation_COW_Eve(*a, **kw):  return _unpack(run_qkd_simulation("COW+Eve", *a, **kw))


# ═══════════════════════════════════════════════════════════════════════
#  Entanglement-based QKD (BBM92 / E91): dedicated topology, shared pipeline
# ═══════════════════════════════════════════════════════════════════════
# Private implementation dispatched by run_qkd_simulation(). It REUSES the
# fork's infrastructure instead of duplicating it:
#   * per-arm FSO loss via the existing _compute_loss (channel_FSO_loss)
#     injected through the fork's QuantumChannel(loss=...) override
#     (attenuation formula is the fallback);
#   * daylight background via the shared _n_background_photons helper
#     (photons per detection gate) folded into a per-round noise probability
#     at each MeasuringNode (accidental coincidences, Kržič Eq. 2.5); dark
#     counts and the gate come from the SAME detector_params contract used
#     by BB84/B92/COW (dark_count [Hz], detection_gate_from_detector);
#   * Eve via the fork's native EveQuantumChannel + generic EveNode on the
#     Charlie -> Bob arm (the SAME photon object is forwarded, so the
#     collapse propagates to Bob's partner photon);
#   * key generation driven by `push(keysize, key_num, run_time)` — the SAME
#     entry point as BB84/B92/COW — so there is NO "number of rounds" knob:
#     the sifted bits of successive emission trains accumulate in `key_bits`
#     and a key is extracted whenever len(key_bits) >= keysize, until the
#     runtime or key_num is exhausted. The total number of emission rounds
#     is a DERIVED quantity, reported as `num_rounds` for diagnostics only;
#   * consequently the secure key rate is estimated by the SHARED
#     `_collect_metrics` (no dedicated metrics block any more): both
#     families run the same code with the same denominator, so R_sk and R_s
#     carry the same unit for all five protocols —
#         R_s  = sifted bits / qubit sent   (send_bits_length per train),
#         R_sk = R_s * [1 - f_EC*H2(E) - H2(E)]   (Kržič Eq. 2.11),
#     i.e. bits per qubit sent. Multiply by `frequency` to recover bits/s
#     (exported separately as the `skr_bits_per_s` diagnostic).
#
# Topology: Charlie (EPS, untrusted) between Alice and Bob; `distance` is the
# TOTAL Alice--Bob separation and `charlie_position` splits it into the two
# arms (see resolve_entanglement_arms), exactly the quantity swept for the
# prepare-and-measure protocols — so the two families are directly
# comparable on every axis.

def _run_entanglement_qkd(
    cfg, ls_params, detector_params, *,
    distance, charlie_position, distance_ac, distance_cb,
    attenuation, loss, loss_parameters, thermal_params,
    polarization_fidelity, source_type,
    eve_intercept_rate, eve_position,
    runtime, keysize, key_num, f_ec, bell_state, seed,
    classical_delay_offset_ps=0.0, enforce_bell_violation=True,
):
    """Entanglement runner (see run_qkd_simulation Observations 3 and 4).

    Returns the same result-dict shape as the prepare-and-measure path
    (qber/throughputs/latency/skr/loss/rs and the per-key ``*_samples``
    lists [+ chsh_S for E91]) so that _worker/_collect_results/_unpack work
    unchanged, plus the raw entanglement diagnostics (num_rounds,
    num_trains, sampled_qber, ...). ``seed`` is the run seed: Alice, Bob,
    Charlie and Eve each get an independent substream from
    :func:`derive_seed`. With ``loss_parameters=None`` and ``loss=None``
    both arms fall back to the attenuation formula, which is what makes the
    family runnable on the attenuation-only reference link as well.
    """
    arm_ac, arm_cb, total_distance = resolve_entanglement_arms(
        distance, charlie_position, distance_ac, distance_cb)

    with_eve = cfg["has_eve"]
    anti = (bell_state == "psi_minus")
    frequency = ls_params["frequency"]
    wavelength_nm = ls_params["wavelength"] * 1e9

    # Same time budget contract as the prepare-and-measure runner.
    tl = Timeline(runtime * 1e9)
    tl.show_progress = False

    # --- per-receiver noise probability (shared detector/thermal contracts) --
    det0 = detector_params[0]
    gate = (thermal_params["detection_gate"] if thermal_params is not None
            else detection_gate_from_detector(det0.get("time_resolution", 1000)))
    n_B = (_n_background_photons(ls_params, thermal_params)
           if thermal_params is not None else 0.0)
    # n_B is a mean photon NUMBER per mode, not a click probability: apply
    # the detector efficiency and the Poisson saturation 1 - exp(-mu).
    det_eff = det0.get("efficiency", 1.0)
    mu_noise = det_eff * n_B + det0.get("dark_count", 0) * gate
    noise_prob = 1.0 - math.exp(-mu_noise)
    pol_err = max(0.0, 1.0 - polarization_fidelity)

    # --- train length: the entanglement analog of the BB84 pulse train ------
    # BB84: light_time = keysize / (frequency * mean_photon_num) and
    #       num_pulses = light_time * frequency = keysize / mean_photon_num.
    # Here one round emits one pulse of the pair source, so the train has
    # exactly the same number of emission attempts for the same keysize.
    mu = (ls_params.get("mean_photon_num")
          if source_type == "eps_poisson" else None)
    rounds_per_train = max(1, int(round(keysize / (mu or 1.0))))

    # --- measuring nodes (hardware + protocol at protocol_stack[0]) ----------
    node_kwargs = dict(qkdtype=cfg["qkdtype"], stack_size=1,
                       detector_efficiency=det0.get("efficiency", 1.0),
                       polarization_error_prob=pol_err,
                       noise_prob_per_round=noise_prob,
                       anti_correlated=anti)
    alice = MeasuringNode("Alice", tl, "alice", rounds_per_train,
                          seed=derive_seed(seed, "ent_alice"), **node_kwargs)
    bob = MeasuringNode("Bob", tl, "bob", rounds_per_train,
                        seed=derive_seed(seed, "ent_bob"), **node_kwargs)

    # --- source node (Charlie, untrusted relay) ------------------------------
    charlie = EntanglementSourceNode("Charlie", tl,
                                     dst_alice="Alice", dst_bob="Bob",
                                     num_rounds=rounds_per_train,
                                     frequency=frequency, seed=derive_seed(seed, "ent_charlie"),
                                     wavelength_nm=wavelength_nm,
                                     bell_state=bell_state,
                                     mean_photon_num=mu)

    # --- quantum channels (fork API: loss override; pol. fidelity = 1.0) -----
    def seg_loss(dist):
        """Loss of one arm, or None to use the channel attenuation formula.

        Args:
            dist (float): arm length [m].

        Returns:
            float | None: loss fraction of that arm.
        """
        if loss_parameters is not None:
            # existing single source of truth for the FSO loss recipe
            return _compute_loss(dist, ls_params, loss_parameters)
        if loss is not None and total_distance > 0:
            # fixed end-to-end loss; split multiplicatively per segment
            return 1.0 - (1.0 - loss) ** (dist / total_distance)
        return None  # channel falls back to the attenuation formula

    qc_common = dict(attenuation=attenuation, frequency=frequency,
                     polarization_fidelity=1.0)

    eve = None
    if with_eve:
        # Fork-native mechanism (same as the prepare-and-measure Eve path):
        # EveQuantumChannel inserts the generic EveNode transparently.
        # `eve_position` is the fraction of the Charlie -> Bob ARM.
        eve = EveNode("Eve", tl, intercept_rate=eve_intercept_rate,
                      seed=derive_seed(seed, "ent_eve"))
        d1 = arm_cb * eve_position
        d2 = arm_cb * (1.0 - eve_position)
        qc_bob = EveQuantumChannel(
            "qc_charlie_bob", tl, eve_node=eve,
            attenuation=attenuation, distance=arm_cb,
            frequency=frequency, eve_position=eve_position,
            loss_seg1=seg_loss(d1), loss_seg2=seg_loss(d2),
            pf_seg1=1.0, pf_seg2=1.0,
        )
    else:
        qc_bob = QuantumChannel("qc_charlie_bob", tl, distance=arm_cb,
                                  loss=seg_loss(arm_cb), **qc_common)
    qc_bob.set_ends(charlie, "Bob")


    qc_alice = QuantumChannel("qc_charlie_alice", tl, distance=arm_ac,
                            loss=seg_loss(arm_ac), **qc_common)
    qc_alice.set_ends(charlie, "Alice")

    # --- classical channels (Alice <-> Bob, over the full separation) --------
    cc_ab = ClassicalChannel("cc_alice_bob", tl, distance=total_distance)
    cc_ab.delay += classical_delay_offset_ps      # same offset as the P&M family
    cc_ab.set_ends(alice, "Bob")
    cc_ba = ClassicalChannel("cc_bob_alice", tl, distance=total_distance)
    cc_ba.delay += classical_delay_offset_ps
    cc_ba.set_ends(bob, "Alice")

    # --- pair the protocols (BB84-style helper from the registry) ------------
    cfg["pair_fn"](alice.protocol_stack[0], bob.protocol_stack[0],
                   anti_correlated=anti)
    alice_proto = alice.protocol_stack[0]
    # simulation-side handle used to size/launch the emission trains
    alice_proto.attach_source(charlie.source)

    # --- schedule: SAME entry point as BB84/B92/COW --------------------------
    tl.init()
    tl.schedule(Event(0, Process(alice_proto, "push",
                                 [keysize, key_num, 6e12])))
    tl.run()

    # --- metrics: the SHARED estimator, no dedicated block -------------------
    # Identical function, identical denominator (send_bits_length = qubits
    # emitted per train) => R_sk and R_s are in bits per qubit sent for the
    # five protocols alike.
    metrics = _collect_metrics(alice_proto, f_ec)
    qber, skr = metrics["qber"], metrics["skr"]

    mean_qber = _safe_mean(qber, default=0.0)
    secure_fraction = max(0.0, 1.0 - f_ec * binary_entropy(mean_qber)
                          - binary_entropy(mean_qber))

    # Channel loss reported end to end (Alice--Bob through Charlie):
    #     L_total = 1 - (1 - L_ac) * (1 - L_cb)
    # `seg_loss` returns None on the reference link (attenuation formula).
    loss_ac, loss_cb = seg_loss(arm_ac), seg_loss(arm_cb)
    if loss_ac is None or loss_cb is None:
        loss_ac = attenuation_loss(arm_ac, attenuation)
        loss_cb = attenuation_loss(arm_cb, attenuation)
    loss_total = 1.0 - (1.0 - loss_ac) * (1.0 - loss_cb)

    result = dict(
        metrics,
        # ── shared metric contract (consumed by _worker/_unpack) ──
        loss=loss_total,
        # ── entanglement diagnostics ──
        protocol=("BBM92" if cfg["qkdtype"] == 3 else "E91"),
        with_eve=with_eve, bell_state=bell_state,
        # `num_rounds` survives as a DERIVED quantity: it is counted, never
        # configured (num_trains * rounds_per_train).
        num_rounds=alice_proto.num_rounds,
        num_trains=alice_proto.num_trains,
        rounds_per_train=rounds_per_train,
        keys_generated=len(alice_proto.error_rates),
        keysize=keysize,
        detected_alice=alice.detected_total, detected_bob=bob.detected_total,
        noise_counts=(alice.noise_counts, bob.noise_counts),
        eve_intercepted=(eve.intercepted_count if eve else 0),
        sampled_qber=_safe_mean(alice_proto.sampled_qbers),
        key_error_rate=mean_qber,
        secure_fraction=secure_fraction, f_ec=f_ec,
        # unit-explicit extras (not part of the shared CSV contract)
        skr_bits_per_s=skr * frequency,
        loss_ac=loss_ac, loss_cb=loss_cb,
        distance_ac=arm_ac, distance_cb=arm_cb, distance=total_distance,
        charlie_position=(arm_ac / total_distance if total_distance else None),
        quantum_time_s=alice_proto.num_rounds / frequency if frequency else 0.0,
        raw_end_time_s=tl.now() * 1e-12,
    )
    if cfg.get("needs_chsh"):
        # Per-train Bell parameters computed by Alice: the list feeds the
        # error bar, its mean is the value published in the CSV.
        result["chsh_S"] = list(alice_proto.chsh_values)
        S = _safe_mean(alice_proto.chsh_values)
        # The E91 security witness is the Bell violation, not the QBER
        # alone: without this guard the CSV would publish R_sk > 0 at |S| <= 2.
        if enforce_bell_violation and (np.isnan(S) or abs(S) <= 2.0):
            result["skr"] = 0.0
            result["skr_samples"] = [0.0] * len(result.get("skr_samples", []))
    return result

# Thin wrappers, symmetric with simulation_BB84 / simulation_B92 / ...:
# each one runs its registry protocol and returns the historical tuple.
def simulation_BBM92(*a, **kw):     return _unpack(run_qkd_simulation("BBM92", *a, **kw))
def simulation_E91(*a, **kw):       return _unpack(run_qkd_simulation("E91", *a, **kw))
def simulation_BBM92_Eve(*a, **kw): return _unpack(run_qkd_simulation("BBM92+Eve", *a, **kw))
def simulation_E91_Eve(*a, **kw):   return _unpack(run_qkd_simulation("E91+Eve", *a, **kw))


# ═══════════════════════════════════════════════════════════════════════
#  Per-task parameter computation
# ═══════════════════════════════════════════════════════════════════════
def _compute_loss(distance, ls_params, loss_parameters):
    """Forward the entire `loss_parameters` dict to channel_FSO_loss.

    ``wind_speed_perp`` is dropped because it is not an argument of
    ``channel_FSO_loss``: it only drives the COW atmospheric phase process.

    Args:
        distance (float): link (or arm) length [m].
        ls_params (dict): light-source parameters (uses ``wavelength``).
        loss_parameters (dict | None): FSO/atmospheric parameters; None on
            the attenuation-only reference link.

    Returns:
        float | None: loss fraction in [0, 1], or None when no atmospheric
        model is configured (the channel then applies its dB/m formula).
    """
    if loss_parameters is None:
        return None
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
    scenario) automatically get them right. With ``loss_parameters=None``
    the loss stays None and the channel falls back to the dB/m formula,
    which is how the attenuation-only reference link is materialised.

    Args:
        distance (float): Alice--Bob separation of this task [m].
        keysize (int): key length requested per key [bits].
        ls_params (dict): light-source parameters of this protocol.
        detector_params (list[dict]): one dict per detector of the receiver.
        loss_parameters (dict | None): FSO/atmospheric parameters.
        thermal_params (dict | None): sky-background parameters.
        runtime (float): simulated time budget [ms].
        attenuation (float): dB/m coefficient of the fallback loss model.
        pfid_initial (float | None): polarisation fidelity; None means 1.0.
        key_num (int): keys generated per point, i.e. Monte Carlo samples.
        source_type (str): source model of this protocol ("sps", "wcp", ...).
        extra_kwargs (dict | None): overlay applied last.

    Returns:
        dict: kwargs ready to be splatted into :func:`run_qkd_simulation`.
    """
    loss = _compute_loss(distance, ls_params, loss_parameters)

    # Turbulence preserves single-mode polarisation (error <1% over 144 km,
    # Fedrizzi et al., Nat. Phys. 5, 389 (2009)), hence the 1.0 default.
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
                 base_keysize, key_num, protocols, global_seed,
                 n_replicas=1, extra_kwargs=None):
    """Build the (point x protocol) task list from a generic point->spec.

    ``point_to_spec(point)`` must return a dict with:
        'sweep_var', 'sweep_val'     (mandatory -- label the output axis)
        'distance', 'keysize'        (optional -- override the defaults)
        'loss_parameters'            (optional -- override base_loss_p)
        'thermal_params'             (optional -- override base_thermal_p)
        'ls_overrides'               (optional -- overlay on protocol ls_params)
        'kwarg_override'             (optional -- overlay on final kwargs)

    Each point x protocol is expanded into ``n_replicas`` tasks, and each
    task receives its own run seed derived from ``global_seed`` and from
    the task identity (protocol, sweep variable, sweep value, replica
    index). Three consequences matter scientifically: re-running the
    campaign with the same global seed reproduces every point exactly;
    neighbouring points/protocols use independent streams instead of
    replaying the same noise realisation; and the replicas of one point
    re-draw the whole run, atmospheric realisation included, which is what
    :func:`replicate_statistics` needs to build an ensemble error bar.


    Args:
        points: iterable of sweep points handed to ``point_to_spec``.
        point_to_spec (callable): maps one point to the spec dict above.
        runtime (float): simulated time budget per task [ms].
        channel_parameters (tuple): (distance [m], attenuation [dB/m],
            polarization_fidelity).
        ls_lookup (dict): light-source params keyed by registry ``ls_key``.
        det_lookup (dict): detector params keyed by registry ``det_key``.
        base_loss_p (dict | None): default FSO parameters.
        base_thermal_p (dict | None): default sky-background parameters.
        base_keysize (int): key size used when the spec does not override it.
        key_num (int): keys generated per replica.
        protocols (list[str]): registry keys to simulate at every point.
        global_seed (int): campaign seed, see :func:`resolve_global_seed`.
        n_replicas (int): independent re-simulations of every point.
        extra_kwargs (dict | None): overlay applied to every task.

    Returns:
        list[dict]: one task dict per (point, protocol, replica) triple.
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
            # Per-protocol override, preserving the detector count of each
            # QSDetector (2 for BB84/B92, 3 for COW); global one otherwise.
            det_by_key = spec.get("detector_params_by_key")
            if det_by_key is not None:
                det_p = det_by_key[cfg["det_key"]]
            else:
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
            
            
            # The task identity fixes the seed, so the point is reproducible
            # regardless of the order in which the pool happens to run it.
            for replica in range(max(1, int(n_replicas))):
                rep_kwargs = dict(kwargs)
                rep_kwargs["seed"] = derive_seed(global_seed, proto, sweep_var,
                                                 sweep_val, replica)
                tasks.append({"protocol": proto, "sweep_var": sweep_var,
                              "sweep_val": sweep_val, "replica": replica,
                              "kwargs": rep_kwargs})

    return tasks


# ═══════════════════════════════════════════════════════════════════════
#  Worker + result wiring (used by every sweep type)
# ═══════════════════════════════════════════════════════════════════════
def _sampled_keys_of(proto):
    """Metric keys carrying per-key samples for a given protocol.

    Args:
        proto (str): registry key of the protocol.

    Returns:
        list[str]: result-dict keys reduced by :func:`replicate_statistics`.
    """
    keys = ["skr", "qber", "throughputs", "rs"]
    if PROTOCOL_REGISTRY[proto]["needs_visibility"]:
        keys.append(_VIS_COL[1])
    if PROTOCOL_REGISTRY[proto].get("needs_chsh"):
        keys.append(_CHSH_COL[1])
    return keys


def _worker(task):
    """Run ONE replica of a sweep point and return its raw per-key samples.

    Reduction to (mean, std, sem) deliberately does NOT happen here: the
    samples of a single replica are clustered by the run's shared
    atmospheric realisation, so they are pooled across replicas by
    :func:`_aggregate_point` before any dispersion is computed.


    Args:
        task (dict): entry produced by :func:`_build_tasks`.

    Returns:
        dict: record with the replica index, the scalar metrics and a
        ``samples`` sub-dict of per-key lists.
    """
    proto = task["protocol"]
    sweep_var = task["sweep_var"]
    sweep_val = task["sweep_val"]
    res = run_qkd_simulation(proto, **task["kwargs"])

    samples = {"skr": res.get("skr_samples"), "qber": res.get("qber"),
               "rs": res.get("rs_samples"),
               "throughputs": res.get("throughput_samples")}

    if "visibility" in res:
        samples[_VIS_COL[1]] = res["visibility"]
    if "chsh_S" in res:
        samples[_CHSH_COL[1]] = res["chsh_S"]
    return {"protocol": proto, sweep_var: sweep_val,
            "replica": task.get("replica", 0),
            "latency": res["latency"], "loss": res["loss"],
            "samples": samples}


def _fallback_result(proto, sweep_var, sweep_val, replica=0):
    """Empty result for a failed replica; keeps the record shape intact.

    Args:
        proto (str): protocol whose task failed.
        sweep_var (str): name of the swept variable.
        sweep_val: value of the swept variable at the failed point.
        replica (int): index of the failed replica.

    Returns:
        dict: record with no samples, so the replica simply does not
        contribute to the statistics of its point.
    """
    return {"protocol": proto, sweep_var: sweep_val, "replica": replica,
            "latency": np.nan, "loss": np.nan,
            "samples": {key: [] for key in _sampled_keys_of(proto)}}


def _aggregate_point(proto, records):
    """Reduce every replica of one sweep point to the published metrics.

    Args:
        proto (str): protocol of the point.
        records (list[dict]): worker outputs for this (protocol, value).

    Returns:
        dict: flat metric record, with ``<key>``/``<key>_std``/``<key>_sem``
        per sampled metric plus ``n_keys`` and ``n_replicas``.
    """
    out = {"latency": _safe_mean([r["latency"] for r in records]),
           "loss": _safe_mean([r["loss"] for r in records])}
    n_keys = n_replicas = 0
    for key in _sampled_keys_of(proto):
        per_replica = [r["samples"].get(key) for r in records]
        mean, std, sem, keys_used, reps = replicate_statistics(per_replica)
        out[key] = mean
        out[f"{key}_std"] = std
        out[f"{key}_sem"] = sem
        # Every metric is measured on the same runs, so the largest count
        # is the one describing the point (a metric can be all-NaN).
        n_keys = max(n_keys, keys_used)
        n_replicas = max(n_replicas, reps)
    out["n_keys"] = n_keys
    out["n_replicas"] = n_replicas
    return out


def _protocol_columns(proto):
    """(label, key) pairs of every CSV column produced for one protocol.

    Sampled metrics expand into three columns (mean, ``_std``, ``_sem``);
    deterministic ones keep a single column. The key count of the point is
    appended so that an error bar can always be traced back to its N.

    Args:
        proto (str): registry key of the protocol.

    Returns:
        list[tuple]: (column label, result-dict key) pairs, in CSV order.
    """
    base = list(_METRIC_COLS)
    if PROTOCOL_REGISTRY[proto]["needs_visibility"]:
        base.append(_VIS_COL)
    if PROTOCOL_REGISTRY[proto].get("needs_chsh"):
        base.append(_CHSH_COL)

    cols = []
    for label, key in base:
        if key in _SAMPLED_METRIC_KEYS:
            cols += [(f"{label}{s}", f"{key}{s}") for s in _STAT_SUFFIXES]
        else:
            cols.append((label, key))
    cols.append(_NKEYS_COL)
    cols.append(_NREPLICAS_COL)
    return cols


def _collect_results(sweep_var, sweep_values, results_list, protocols,
                     global_seed=None):
    """Reorganise raw per-task results into wide-format columns.

    Args:
        sweep_var (str): name of the swept variable (first CSV column).
        sweep_values (list): swept values, in the order they must appear.
        results_list (list[dict]): worker outputs, in arbitrary order.
        protocols (list[str]): protocols expected in the output.
        global_seed (int | None): campaign seed recorded as a constant
            column, so a detached CSV still documents how to reproduce it.

    Returns:
        dict: column name -> np.ndarray, ready for ``pd.DataFrame``.
    """
    # Group the replicas of each (protocol, sweep value) before reducing.
    grouped = {p: {} for p in protocols}

    for r in results_list:
        if r["protocol"] in grouped:
            grouped[r["protocol"]].setdefault(r[sweep_var], []).append(r)
    data = {p: {v: _aggregate_point(p, recs) for v, recs in points.items()}
            for p, points in grouped.items()}

    metrics = {sweep_var: np.array(sweep_values)}
    if global_seed is not None:
        metrics["global_seed"] = np.full(len(sweep_values), global_seed)
    for proto in protocols:
        for label, key in _protocol_columns(proto):
            metrics[f"{label}-{proto}"] = np.array(
                [data[proto].get(v, {}).get(key, np.nan) for v in sweep_values])
    return metrics


def _run_tasks(tasks, label, sweep_values, protocols, output_csv, max_workers,
               global_seed=None):
    """Run the task list in parallel and save the wide-format metrics CSV.

    Args:
        tasks (list[dict]): output of :func:`_build_tasks`.
        label (str): name of the swept variable (first CSV column).
        sweep_values (list): swept values, in output order.
        protocols (list[str]): protocols expected in the output.
        output_csv (str): destination path; parent directories are created.
        max_workers (int | None): pool size; defaults to the CPU count.
        global_seed (int | None): campaign seed recorded in the CSV.

    Returns:
        dict: the metrics table, also written to ``output_csv``.

    Raises:
        ValueError: if ``label`` collides with a reserved metric key.
    """
    if label in _RESERVED_RESULT_KEYS:
        raise ValueError(
            f"sweep_var/label {label!r} collides with a reserved metric key "
            f"({sorted(_RESERVED_RESULT_KEYS)}); rename the sweep "
            f"(e.g. atmospheric visibility -> 'atm_visibility').")
    if max_workers is None:
        max_workers = os.cpu_count() or 4

    total = len(tasks)
    n_replicas = max(1, total // max(1, len(sweep_values) * len(protocols)))
    print(f"[parallel] Launching {total} tasks across {max_workers} workers "
          f"({len(sweep_values)} {label}s x {len(protocols)} protocols "
          f"x {n_replicas} replicas)")

    results = []
    t_start = time.time()
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_worker, t): t for t in tasks}
        pending, i = set(futures), 0
        while pending:
            # Heartbeat: wake up every 30 s even with no task finished, so
            # the user can tell "working" apart from "stuck".
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
                    rep = t.get("replica", 0)
                    print(f"\n[parallel] WARNING: {proto} @ {label}={val} "
                          f"(replica {rep}) failed: {exc}")
                    results.append(_fallback_result(proto, label, val, rep))
                print(f"\r[parallel] {i}/{total} done ({i/total*100:.1f}%), "
                      f"elapsed {time.time() - t_start:.0f}s",
                      end="", flush=True)


    print()

    metrics = _collect_results(label, sweep_values, results, protocols,
                               global_seed=global_seed)
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
                 max_workers=None, extra_kwargs=None,
                 site=None, global_seed=None, output_dir=DEFAULT_DATA_DIR,
                 n_replicas=1):
    """Parallel sweep over a single kwarg of run_qkd_simulation.

    Args:
        sweep_var: name of the parameter to sweep. Must be a kwarg of
            run_qkd_simulation, e.g. 'distance', 'keysize', 'attenuation',
            'polarization_fidelity', 'eve_intercept_rate', ...
        sweep_values: iterable of values to assign to `sweep_var`.
        runtime: simulated time budget per point [ms].
        channel_parameters: (distance [m], attenuation [dB/m],
            polarization_fidelity) of the fixed operating point.
        ls_params / ls_params_cow: light-source parameters per family.
        detector_params / detector_params_cow: detector parameter lists.
        key_num: keys generated per replica. Consecutive keys advance the
            RNG stream, so they differ, but they share the run's atmospheric
            realisation; they measure the WITHIN-run variation.
        loss_parameters: FSO/atmospheric parameters, or None to simulate the
            attenuation-only reference link.
        thermal_params: sky-background parameters, or None for no background.
        keysize: key length used when the sweep does not override it.
        protocols: subset of PROTOCOL_REGISTRY keys (default: all).
        output_csv: where to save results. Default:
            <output_dir>/metrics_variable-<sweep_var>.csv
        max_workers: size of the process pool (default: CPU count).
        extra_kwargs: dict forwarded to every run_qkd_simulation call.
        site: site context used to recompute u* and C_n2 on the sweeps that
            change T / P / height / wind.
        global_seed: campaign seed; every task derives its own substream
            from it (see :func:`_build_tasks`).
        output_dir: directory of the default ``output_csv``; the realistic
            and reference-link campaigns use different ones so both sets of
            CSVs can coexist.
        n_replicas: independent re-simulations of every point, each with its
            own derived seed. Unlike extra keys, a replica re-draws the
            atmospheric piston and Eve's pattern too, so n_replicas >= 2
            promotes the error bar from within-run to ensemble level (see
            :func:`replicate_statistics`). Cost scales as key_num*n_replicas.


    Returns:
        dict of column_name -> np.ndarray (also saved as CSV).

    Raises:
        ValueError: if the sweep is unknown, or if it needs an atmospheric
            or thermal model that this campaign does not configure.
    """
    if protocols is None:
        protocols = list(PROTOCOL_REGISTRY.keys())
    if output_csv is None:
        output_csv = os.path.join(output_dir, f"metrics_variable-{sweep_var}.csv")
    sweep_values = list(sweep_values)
    global_seed = resolve_global_seed(global_seed)

    ls_lookup  = {"ls_params": ls_params, "ls_params_cow": ls_params_cow}
    det_lookup = {"detector_params": detector_params,
                  "detector_params_cow": detector_params_cow}

    # Sweeps that reshape the atmosphere/sky are meaningless when those
    # models are absent (attenuation-only reference link).
    if loss_parameters is None and sweep_var in _ATMOSPHERIC_SWEEPS:
        raise ValueError(
            f"sweep {sweep_var!r} requires loss_parameters; it is not "
            "available on the attenuation-only reference link.")
    if thermal_params is None and sweep_var in _THERMAL_SWEEPS:
        raise ValueError(
            f"sweep {sweep_var!r} requires thermal_params; it is not "
            "available on the attenuation-only reference link.")

    # Site context needed to RECOMPUTE the derived quantities (u*, C_n2)
    # when T / P / height / wind are swept.
    site = site or {}
    site_altitude_for_sweeps = site.get("site_altitude", 0.0)
    base_ground_wind = site.get("wind_speed", 3.2)
    cn2_reference_hour = site.get("cn2_reference_hour", 12.0)
    cn2_sunrise = site.get("sunrise", 6.0)
    cn2_sunset = site.get("sunset", 18.0)
    cn2_relative_humidity = site.get("relative_humidity", 47.0)


    def point_to_spec(val):
        """Materialise the parameter set of one point of this sweep.

        Args:
            val: value taken by ``sweep_var`` at this point.

        Returns:
            dict: spec consumed by :func:`_build_tasks`.

        Raises:
            ValueError: if ``sweep_var`` is not a supported sweep.
        """
        spec = {
            "sweep_var": sweep_var,
            "sweep_val": val
        }

        # Copies, so that the base parameter dicts are never mutated.
        lp = dict(loss_parameters) if loss_parameters is not None else None
        tp = dict(thermal_params) if thermal_params is not None else None
        ls_overrides = {}
        det_override = None

        def _override_detectors(key, value):
            """Apply ``key=value`` to EVERY detector of EVERY protocol.

            The override is built per registry ``det_key`` so that each
            QSDetector keeps its own detector count (2 for BB84/B92, 3 for
            COW). Deriving one single list from ``detector_params`` would
            leave the COW DM2 detector at the library default, biasing the
            monitoring-line visibility that feeds R_sk^COW.

            Args:
                key (str): detector parameter to override.
                value: value assigned to that parameter.
            """
            spec["detector_params_by_key"] = {
                dk: [{**d, key: value} for d in dl]
                for dk, dl in det_lookup.items()
            }

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
            "f_ec",
            # entanglement-only kwargs of run_qkd_simulation (BBM92/E91).
            # NOTE: there is no "num_rounds" knob any more -- the emission
            # rounds are derived from `keysize` (sweep "keysize" instead).
            "charlie_position",
            "distance_ac",
            "distance_cb",
            "bell_state",
        ]:
            spec["kwarg_override"] = {sweep_var: val}

        elif sweep_var == "frequency":
            ls_overrides["frequency"] = val
        elif sweep_var == "mean_photon_num":
            ls_overrides["mean_photon_num"] = val
        elif sweep_var == "efficiency":
            _override_detectors("efficiency", val)
        elif sweep_var == "dark_count":
            _override_detectors("dark_count", val)
        elif sweep_var == "atm_visibility":
            lp["atm_visibility"] = val
        elif sweep_var == "C_n2":
            lp["C_n2"] = val
        elif sweep_var in ("temperature", "pressure", "height_ag",
                           "ground_wind_speed"):
            # T, P, height and wind are not independent inputs of
            # channel_FSO_loss: they also set u* and C_n2, recomputed below.
            if sweep_var == "ground_wind_speed":
                lp["wind_speed_perp"] = wind_speed_perp(
                    site_altitude_for_sweeps, val)
                ground_wind = val
            else:
                lp[sweep_var] = val
                ground_wind = base_ground_wind
            lp["friction_velocity"] = f_velocity(
                ground_wind, T_classification=7,
                height_ag=lp["height_ag"])
            lp["C_n2"] = cn2_horizontal_link(
                lp["height_ag"], hour=cn2_reference_hour,
                sunrise=cn2_sunrise, sunset=cn2_sunset,
                temperature=lp["temperature"], wind_speed=ground_wind,
                relative_humidity=cn2_relative_humidity)
        elif sweep_var == "wind_speed_perp":
            # Acts only on the COW atmospheric phase process, so 9 of the 10
            # protocols stay flat; prefer the "ground_wind_speed" sweep.
            lp["wind_speed_perp"] = val
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
            raise ValueError(f"Sweep '{sweep_var}' is not supported.")

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
        protocols=protocols, global_seed=global_seed,
        n_replicas=n_replicas, extra_kwargs=extra_kwargs,
    )
    return _run_tasks(tasks, sweep_var, sweep_values, protocols,
                      output_csv, max_workers, global_seed=global_seed)


def sim_scenario(label, scenario_points, *, runtime, channel_parameters,
                 ls_params, ls_params_cow, detector_params, detector_params_cow,
                 keysize, key_num, base_loss_parameters, base_thermal_params,
                 diurnal_profile_fn, protocols=None, output_csv=None,
                 max_workers=None, extra_kwargs=None, global_seed=None,
                 output_dir=DEFAULT_DATA_DIR, n_replicas=1):
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
        global_seed: campaign seed; each task derives its own substream.
        output_dir: directory of the default ``output_csv``.
        n_replicas: independent re-simulations of every point, as in
            :func:`sim_variable`.
        ... (other args as in sim_variable)

    Returns:
        dict of column_name -> np.ndarray (also saved as CSV).
    """
    if protocols is None:
        protocols = list(PROTOCOL_REGISTRY.keys())
    if output_csv is None:
        output_csv = os.path.join(output_dir, f"metrics_scenario-{label}.csv")
    scenario_points = list(scenario_points)
    global_seed = resolve_global_seed(global_seed)

    ls_lookup  = {"ls_params": ls_params, "ls_params_cow": ls_params_cow}
    det_lookup = {"detector_params": detector_params,
                  "detector_params_cow": detector_params_cow}

    def point_to_spec(token):
        """Expand one scenario token into a full parameter set.

        Args:
            token: opaque point (e.g. the hour of the day).

        Returns:
            dict: spec consumed by :func:`_build_tasks`.
        """
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
        protocols=protocols, global_seed=global_seed,
        n_replicas=n_replicas, extra_kwargs=extra_kwargs,
    )
    return _run_tasks(tasks, label, scenario_points, protocols,
                      output_csv, max_workers, global_seed=global_seed)


# ═══════════════════════════════════════════════════════════════════════
#  Execution driver
# ═══════════════════════════════════════════════════════════════════════
def default_environment():
    """Single source of truth for the simulation environment.

    Builds the light-source, detector, atmospheric/site, FSO-loss and
    thermal-noise parameter sets used by ALL drivers (run_simulation for
    BB84/B92/COW and run_entanglement_simulation for BBM92/E91), so the
    hardware/site configuration is defined exactly once.

    Returns:
        dict with keys: ls_params, ls_params_cow, detector_params,
        detector_params_cow, channel_parameters, loss_parameters,
        thermal_params, site (sunrise/sunset/latitude/longitude/altitude...),
        extra_kwargs and common (ready-to-use kwargs for sim_variable).
    """
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
    #  Version 1.2. Thorlabs GmbH, 2023. Document MTN028160-D02.'
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
    site_altitude = 720.0             # m ASL - only for wind_speed_perp (jet)
    latitude      = math.radians(-23.5615)   # LARC/EPUSP
    longitude     = math.radians(-46.7311)
    sunrise = 6.7833
    sunset  = 19.8833         # Sunset/sunrise on Feb 04, 2015 (twilight type -0.833 deg)

    friction_velocity = f_velocity(wind_speed, T_classification=7, height_ag=height_link)      # m/s
    viscosity = viscosity_sutherland(temperature)

    cn = cn2_horizontal_link(height_link, hour=12.0, sunrise=sunrise,
                             sunset=sunset, temperature=temperature,
                             wind_speed=wind_speed, relative_humidity=47.0)

    # -- (distance_m, attenuation_dB/m, polarization_fidelity)
    channel_parameters = (700, 0.0002, 1)

    # -- FSO loss parameters (forwarded as **kwargs to channel_FSO_loss)
    loss_parameters = {
        "atm_visibility":     10e3,             # m   (Measured using 'WORLD METEOROLOGICAL ORGANIZATION. Guide to Instruments and Methods of Observation. 2024. p. 352–374')
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

    extra_kwargs = None

    # -- Common kwargs reused by every variable-sweep.
    # The Monte Carlo sample size of a point is key_num * n_replicas. The
    # historical key_num=1 with a single run gave one draw, no mean and no
    # error bar, hiding every small-effect sweep (C_n2, T, P, height_ag,
    # wind, charlie_position, dark_count) under a ~5% spread in R_sk. The
    # replicas are what make the error bar an ENSEMBLE quantity, since they
    # re-draw the atmospheric realisation that the keys of one run share.

    common = dict(
        runtime=1000,
        channel_parameters=channel_parameters,
        ls_params=ls_params, ls_params_cow=ls_params_cow,
        detector_params=detector_params,
        detector_params_cow=detector_params_cow,
        key_num=KEY_NUM_FOR_STATISTICS,
        n_replicas=N_REPLICAS_FOR_STATISTICS,
        loss_parameters=loss_parameters,
        thermal_params=thermal_params,
        extra_kwargs=extra_kwargs,
    )

    site = dict(temperature=temperature, pressure=pressure,
                wind_speed=wind_speed, height_link=height_link,
                site_altitude=site_altitude, latitude=latitude,
                longitude=longitude, sunrise=sunrise, sunset=sunset,
                relative_humidity=47.0, cn2_reference_hour=12.0)
    # `site` is also consumed by sim_variable to RECOMPUTE u* and C_n2 when
    # T / P / height / wind are swept (see point_to_spec).
    common["site"] = site

    return dict(ls_params=ls_params, ls_params_cow=ls_params_cow,
                detector_params=detector_params,
                detector_params_cow=detector_params_cow,
                channel_parameters=channel_parameters,
                loss_parameters=loss_parameters,
                thermal_params=thermal_params,
                site=site, extra_kwargs=extra_kwargs, common=common)


def reference_link_environment():
    """Environment of the attenuation-only reference link (commit 11ad36f).

    This is the baseline the realistic free-space campaign is compared
    against. It reuses the SAME hardware as :func:`default_environment`
    (laser, detectors, source types), so the only difference between the
    two campaigns is the CHANNEL:

    ================  =========================  ========================
    quantity          realistic aerial link      reference link
    ================  =========================  ========================
    loss              channel_FSO_loss (fog,     1 - 10**(-alpha*d/10)
                      turbulence, rain)
    background        sky radiance B_sky         none
    channel phase     atmospheric piston (COW)   none
    pol. fidelity     1.0                        0.97 (historical value)
    ================  =========================  ========================

    ``loss_parameters`` and ``thermal_params`` are therefore None, which is
    exactly what makes :func:`_build_task_kwargs` fall back to the dB/m
    formula and skip the thermal source.

    Returns:
        dict: same shape as :func:`default_environment` (minus the
        atmospheric entries), including a ready-to-use ``common`` dict.
    """
    env = default_environment()

    # Historical operating point of 11ad36f: 700 m, 2e-4 dB/m, pfid 0.97.
    channel_parameters = (700, 0.0002, 0.97)

    common = dict(
        runtime=1000,
        channel_parameters=channel_parameters,
        ls_params=env["ls_params"], ls_params_cow=env["ls_params_cow"],
        detector_params=env["detector_params"],
        detector_params_cow=env["detector_params_cow"],
        key_num=KEY_NUM_FOR_STATISTICS,
        n_replicas=N_REPLICAS_FOR_STATISTICS,
        loss_parameters=None,
        thermal_params=None,
        extra_kwargs=None,
        site=env["site"],
        output_dir=REFERENCE_LINK_DATA_DIR,
    )
    return dict(env, channel_parameters=channel_parameters,
                loss_parameters=None, thermal_params=None,
                extra_kwargs=None, common=common)


def build_diurnal_profile_fn(env):
    """Diurnal profile bound to the site of `env` (used by both drivers)."""
    site = env["site"]
    df = pd.read_csv("sensores/estação-solar-usp_Tabela01.dat", sep=',', skiprows=4, header=None, decimal='.', low_memory=False)
    df[0] = pd.to_datetime(df[0], format="%Y-%m-%d %H:%M:%S")
    return partial(diurnal_profile, sunrise=site["sunrise"], sunset=site["sunset"],
                   site_altitude=site["site_altitude"], latitude=site["latitude"],
                   longitude=site["longitude"],
                   local_tz=timezone(timedelta(hours=-2)), date="2015-02-04",
                   dataframe=df)


def save_simulator_metrics(path, elapsed_s, global_seed, key_num,
                           campaigns, n_replicas=1, extra=None):
    """Record the campaign metadata needed to replicate the run.

    Storing the global seed is what makes the results reproducible in a
    scientific sense: re-running with ``QKD_GLOBAL_SEED=<seed>`` (or
    ``run_simulation(global_seed=<seed>)``) replays every point of every
    sweep exactly, because each task seed is derived deterministically from
    it by :func:`derive_seed`.

    Args:
        path (str): destination CSV.
        elapsed_s (float): wall-clock duration of the campaign [s].
        global_seed (int): seed that generated the whole campaign.
        key_num (int): keys generated per replica.
        campaigns (str): which link models were simulated.
        n_replicas (int): independent re-simulations per point; the total
            Monte Carlo sample size of a point is key_num * n_replicas.
        extra (dict | None): additional single-value columns.

    Returns:
        str: the path written.
    """
    record = {
        "Total_execution_time_(seconds)": [elapsed_s],
        "global_seed": [global_seed],
        "seed_env_var": [GLOBAL_SEED_ENV_VAR],
        "key_num": [key_num],
        "n_replicas": [n_replicas],
        "campaigns": [campaigns],
        "timestamp_utc": [datetime.now(timezone.utc).isoformat(timespec="seconds")],
    }
    for name, value in (extra or {}).items():
        record[name] = [value]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    pd.DataFrame(record).to_csv(path, index=False)
    print(f"[seed] global_seed={global_seed} recorded in {path}")
    return path


def run_reference_link_simulation(global_seed=None):
    """Rerun the reference sweeps on the attenuation-only link of 11ad36f.

    Reproduces the historical campaign (distance sweep from 1 km to 100 km
    in 1 km steps and the same key-size list) with the current pipeline, so
    the five protocols are measured on a channel WITHOUT any atmospheric
    model. Compared point by point with the CSVs of
    :func:`run_simulation`, this isolates the penalty imposed by the
    realistic aerial link (fog/aerosols, turbulence, rain, sky background)
    from the intrinsic behaviour of each protocol.

    Results go to :data:`REFERENCE_LINK_DATA_DIR` so both campaigns coexist.

    Args:
        global_seed (int | None): campaign seed; resolved by
            :func:`resolve_global_seed` when None.

    Returns:
        int: the global seed actually used.
    """
    global_seed = resolve_global_seed(global_seed)
    env = reference_link_environment()
    common = env["common"]
    # Same post-processing offset and source placement as the realistic
    # campaign, so only the channel model differs between the two.
    common["extra_kwargs"] = {"f_ec": 1.0, "charlie_position": 0.1,
                              "classical_delay_offset_ps": 1e9}
    common["global_seed"] = global_seed

    print(f"[campaign] attenuation-only reference link -> "
          f"{REFERENCE_LINK_DATA_DIR}/")
    sim_variable("distance", range(1000, 100001, 1000),
                 keysize=10000, **common)
    sim_variable("keysize",
                 [20, 45, 50, 100, 200, 400, 800, 1600,
                  5000, 20000, 40000, 80000, 100000],
                 **common)
    return global_seed


def run_simulation(global_seed=None, run_reference_link=True):
    """Run the full comparison campaign and record its metadata.

    Executes the realistic free-space (aerial link) sweeps and then, unless
    disabled, the attenuation-only reference campaign of
    :func:`run_reference_link_simulation`, so the two link models can be
    compared. The global seed and the sample size behind the error bars are
    saved to ``data/simulator_metrics.csv``.

    Args:
        global_seed (int | None): campaign seed; when None it comes from
            ``$QKD_GLOBAL_SEED`` or :data:`DEFAULT_GLOBAL_SEED`.
        run_reference_link (bool): also run the reference-link campaign.

    Returns:
        int: the global seed actually used.
    """
    start = time.time()
    global_seed = resolve_global_seed(global_seed)

    env = default_environment()
    common = env["common"]
    common["global_seed"] = global_seed
    # Explicit charlie_position: the signature default applies to EVERY
    # sweep, and 0.1 (source at 10% of the link) matches the asymmetric
    # configuration used by Krzic rather than the symmetric README case.
    common["extra_kwargs"] = {**(common["extra_kwargs"] or {}), "f_ec": 1.0,
                              "charlie_position": 0.1,
                              "classical_delay_offset_ps": 1e9}
    site_altitude = env["site"]["site_altitude"]
    print(f"[campaign] realistic free-space aerial link -> "
          f"{DEFAULT_DATA_DIR}/  (global_seed={global_seed})")

    # `distance` is the total Alice--Bob separation for EVERY protocol; for
    # BBM92/E91 the source sits at `charlie_position` along that link.
    sim_variable("distance", range(100, 2001, 100),
                 keysize=10000, **common)

    #sim_variable("distance", range(1000, 100001, 1000),
    #             keysize=10000, **common)

    # Single keysize sweep for BOTH families: the entanglement protocols are
    # keysize-oriented too (the emission rounds are derived from it).
    sim_variable("keysize",
                 [20, 45, 50, 100, 200, 400, 800, 1600,
                  5000, 20000, 40000, 80000, 100000],
                 **common)

    # Where the untrusted source sits along the Alice--Bob link.
    sim_variable("charlie_position",
                 [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                 keysize=10000, protocols=ENTANGLEMENT_PROTOCOLS, **common)

    sim_variable("f_ec",
                 [1.0,1.05,1.10,1.11,1.12,1.13,1.14,1.15,1.16,1.17,1.18,1.20,1.22],
                 **common)

    sim_variable("eve_intercept_rate", [0.1, 0.3, 0.5, 0.7, 0.9], keysize=10_000, protocols=["BB84+Eve", "B92+Eve", "COW+Eve", "BBM92+Eve", "E91+Eve"], **common)
                 
    sim_variable("efficiency", [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0], keysize=10000, **common)
    # Range widened: at midday B_sky the background is n_B = 8.6e-2
    # photons/gate, three to six orders of magnitude above dark_count*gate.
    sim_variable("dark_count", [1e2,1e3,1e4,1e5,1e6,1e7,1e8], keysize=10000, **common)
    sim_variable("frequency", [1e6,2e6,5e6,8e6,10e6,20e6,50e6], keysize=10000, **common)
    sim_variable("atm_visibility", [100,200,500,1000,2000,5000,10000,20000,50000], keysize=10000, **common)
    sim_variable("C_n2", [1e-18,1e-17,3e-17,1e-16,3e-16,1e-15], keysize=10000, **common)
    sim_variable("temperature", [273,282,293,303,308,313], keysize=10000, **common)
    sim_variable("pressure", [80000,85000,90000,92700, 95000,100000], keysize=10000, **common)

    # Surface wind: propagated to u* AND to C_n2 (see point_to_spec); the
    # older "wind_speed_perp" sweep was inert for 9 of the 10 protocols.
    sim_variable("ground_wind_speed", [0.1,2,5,10,15,20], keysize=10000, **common)
    sim_variable("height_ag", [2,5,8,10,20,50], keysize=10000, **common)
    sim_variable("receiver_radius", [0.025,0.05,0.075,0.10,0.15], keysize=10000, **common)
    sim_variable("filter_bandwidth", [0.1e-9,0.2e-9,0.5e-9,1e-9,2e-9,5e-9,10e-9], keysize=10000, **common)
    sim_variable("fov_solid_angle", [1e-11,1e-10,1e-9,1e-8,1e-7], keysize=10000, **common)
    
    mmh = 2.7778e-7

    sim_variable("precipitation_rate", [0.1*mmh, 1*mmh, 5*mmh, 10*mmh, 20*mmh, 30*mmh], keysize=10000, **common)
    sim_variable("interferometer_phase_error", [0,0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,10.0,100.0], keysize=10000, protocols=["COW","COW+Eve"], **common)
    # eve_position: fraction of the Alice--Bob link for the P&M protocols and
    # of the Charlie->Bob ARM for the entanglement ones (that is the arm Eve
    # attacks; see _run_entanglement_qkd).
    sim_variable("eve_position", [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9], keysize=10000, protocols=["BB84+Eve","B92+Eve","COW+Eve", "BBM92+Eve", "E91+Eve"], **common)

    # -- Sweep #n: scenario by hour of day (site bound by the shared builder).
    diurnp_fn = build_diurnal_profile_fn(env)

    sim_scenario(
        label="hour",
        scenario_points=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
        runtime=1000,
        channel_parameters=env["channel_parameters"],
        ls_params=env["ls_params"], ls_params_cow=env["ls_params_cow"],
        detector_params=env["detector_params"],
        detector_params_cow=env["detector_params_cow"],
        keysize=10_000, key_num=KEY_NUM_FOR_STATISTICS,
        n_replicas=N_REPLICAS_FOR_STATISTICS,
        base_loss_parameters=env["loss_parameters"],
        base_thermal_params=env["thermal_params"],
        diurnal_profile_fn=diurnp_fn,
        extra_kwargs=common["extra_kwargs"],
        global_seed=global_seed,
    )

    campaigns = "free_space"
    if run_reference_link:
        run_reference_link_simulation(global_seed)
        campaigns = "free_space+reference_link"

    save_simulator_metrics(
        os.path.join(DEFAULT_DATA_DIR, "simulator_metrics.csv"),
        elapsed_s=time.time() - start,
        global_seed=global_seed,
        key_num=KEY_NUM_FOR_STATISTICS,
        n_replicas=N_REPLICAS_FOR_STATISTICS,
        campaigns=campaigns,
        extra={"reference_link_dir": REFERENCE_LINK_DATA_DIR},
    )
    return global_seed


if __name__ == "__main__":
    run_simulation()
