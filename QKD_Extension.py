import math
import time

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from matplotlib import pyplot as plt

from sequence.components.optical_channel import QuantumChannel, ClassicalChannel, EveQuantumChannel
from sequence.kernel.event import Event
from sequence.kernel.process import Process
from sequence.kernel.timeline import Timeline
from sequence.qkd.BB84 import pair_bb84_protocols
from sequence.qkd.B92 import pair_b92_protocols
from sequence.qkd.COW import pair_cow_protocols
from sequence.topology.node import QKDNode, EveNode
from sequence.utils.encoding_cow import time_bin_cow
from QCLoss.loss import channel_FSO_loss
import sequence.utils.log as log
import numpy as np
import pandas as pd

def binary_entropy(Q):
    Q = max(0.0, min(1.0, Q))
    if Q == 0 or Q == 1:
        return 0
    return -Q * math.log2(Q) - (1 - Q) * math.log2(1 - Q)
    
def _collect_metrics(protocol, distance: float, attenuation: float):
    """Extrai QBER, throughput, latência, SKR e perda de um protocolo."""
    secret_key_rate_mean = 0

    QBER = protocol.error_rates
    THROUGHPUTS = np.mean(protocol.throughputs) if len(protocol.throughputs) > 0 else 0.0
    LATENCY = protocol.latency
    LOSS = 1-10**((distance*attenuation)/(-10))
    R_s_list = []
    
    if not QBER or protocol.send_bits_length == 0:
        return QBER, THROUGHPUTS, LATENCY, 0.0, LOSS, 0.0
    
    for i, e in enumerate(QBER):
        R_s = protocol.sifted_bits_length[i] / protocol.send_bits_length
        secret_key_rate_mean += max(0.0, R_s * (1 - 2*binary_entropy(e)))
        R_s_list.append(R_s)
    SECRET_KEY_RATE = secret_key_rate_mean/len(QBER)
    mean_R_s = np.mean(np.array(R_s_list, dtype=float))
    
    return QBER, THROUGHPUTS, LATENCY, SECRET_KEY_RATE, LOSS, mean_R_s
    
def _collect_cow_metrics(protocol, visibility, ls_params, distance: float, attenuation: float):
    """Extrai métricas COW com SKR ajustado por visibilidade do monitoramento."""
    secret_key_rate_mean = 0

    qber_list = protocol.error_rates
    throughputs = np.mean(protocol.throughputs) if len(protocol.throughputs) > 0 else 0.0
    latency = protocol.latency
    
    t = 10 ** ((distance * attenuation) / (-10))
    loss = 1 - t
    
    if not qber_list or protocol.send_bits_length == 0:
        return qber_list, throughputs, latency, 0.0, loss, 0.0
    
    # R_sk calculated based on https://doi.org/10.1063/1.2126792
    
    
    mean_photon_num = ls_params["mean_photon_num"]
    r = mean_photon_num*(1-t)
    R_s_list = []

    for i, e in enumerate(qber_list):
        R_s = protocol.sifted_bits_length[i] / protocol.send_bits_length
        
        v = visibility[i]

        eve_info = r + ((1 - v)*(1 + math.exp(-mean_photon_num*t))/(2*math.exp(-mean_photon_num*t)))
        secret_key_rate_mean += max(0.0, R_s * (1 - binary_entropy(e) - eve_info)) 
        R_s_list.append(R_s)
        
    secret_key_rate = secret_key_rate_mean / len(qber_list)
    mean_R_s = np.mean(np.array(R_s_list, dtype=float))
    
    return qber_list, throughputs, latency, secret_key_rate, loss, mean_R_s



def simulation_BB84(ls_params, detector_params, runtime=20, log_filename=-1, distance=1e3, polarization_fidelity=0.97, attenuation=0.0002, keysize=256, key_num=math.inf, source_type="wcp", loss=None):
    tl = Timeline(runtime*1e9)
    tl.show_progress = False

    # set log
    if (log_filename != -1):
        log.set_logger(__name__, tl, log_filename)
        log.set_logger_level('DEBUG')
        log.track_module('BB84')
        #log.track_module('timeline')
        log.track_module('light_source')

    qc0 = QuantumChannel("qc0", tl, distance=distance, polarization_fidelity=polarization_fidelity, attenuation=attenuation, loss=loss)
    qc1 = QuantumChannel("qc1", tl, distance=distance, polarization_fidelity=polarization_fidelity, attenuation=attenuation, loss=loss)
    cc0 = ClassicalChannel("cc0", tl, distance=distance)
    cc1 = ClassicalChannel("cc1", tl, distance=distance)
    cc0.delay += 1e9  # 1 ms
    cc1.delay += 1e9

    # Alice
    alice = QKDNode("alice", tl, stack_size=1, source_type=source_type)
    alice.set_seed(0)

    for name, param in ls_params.items():
        alice.update_lightsource_params(name, param)

    # Bob
    bob = QKDNode("bob", tl, stack_size=1, source_type=source_type)
    bob.set_seed(1)

    for i in range(len(detector_params)):
        for name, param in detector_params[i].items():
            bob.update_detector_params(i, name, param)

    qc0.set_ends(alice, bob.name)
    qc1.set_ends(bob, alice.name)
    cc0.set_ends(alice, bob.name)
    cc1.set_ends(bob, alice.name)

    # BB84 config
    pair_bb84_protocols(alice.protocol_stack[0], bob.protocol_stack[0])
    process = Process(alice.protocol_stack[0], "push", [keysize, key_num, 6e12])
    event = Event(0, process)
    tl.schedule(event)

    tl.init()
    tl.run()

    return _collect_metrics(alice.protocol_stack[0], distance, attenuation)

def simulation_B92(ls_params, detector_params, runtime=20, log_filename=-1, distance=1e3, polarization_fidelity=0.97, attenuation=0.0002, keysize=256, key_num=math.inf, source_type="wcp", loss=None):
    tl = Timeline(runtime*1e9)
    tl.show_progress = False

    # set log
    if (log_filename != -1):
        log.set_logger(__name__, tl, log_filename)
        log.set_logger_level('DEBUG')
        log.track_module('B92')
        #log.track_module('timeline')
        log.track_module('light_source')

    qc0 = QuantumChannel("qc0", tl, distance=distance, polarization_fidelity=polarization_fidelity, attenuation=attenuation, loss=loss)
    qc1 = QuantumChannel("qc1", tl, distance=distance, polarization_fidelity=polarization_fidelity, attenuation=attenuation, loss=loss)
    cc0 = ClassicalChannel("cc0", tl, distance=distance)
    cc1 = ClassicalChannel("cc1", tl, distance=distance)
    cc0.delay += 1e9  # 1 ms
    cc1.delay += 1e9

    # Alice
    alice = QKDNode("alice", tl, stack_size=1, qkdtype=1, source_type=source_type)
    alice.set_seed(0)

    for name, param in ls_params.items():
        alice.update_lightsource_params(name, param)

    # Bob
    bob = QKDNode("bob", tl, stack_size=1, qkdtype=1, source_type=source_type)
    bob.set_seed(1)

    for i in range(len(detector_params)):
        for name, param in detector_params[i].items():
            bob.update_detector_params(i, name, param)

    qc0.set_ends(alice, bob.name)
    qc1.set_ends(bob, alice.name)
    cc0.set_ends(alice, bob.name)
    cc1.set_ends(bob, alice.name)

    # B92 config
    pair_b92_protocols(alice.protocol_stack[0], bob.protocol_stack[0])
    process = Process(alice.protocol_stack[0], "push", [keysize, key_num, 6e12])
    event = Event(0, process)
    tl.schedule(event)

    tl.init()
    tl.run()

    return _collect_metrics(alice.protocol_stack[0], distance, attenuation)


def simulation_COW(ls_params, detector_params, runtime=20, log_filename=-1, distance=1e3, polarization_fidelity=0.97, attenuation=0.0002, keysize=256, key_num=math.inf, phase_noise_coefficient=0.01, interferometer_phase_error=0.20, loss=None):
    tl = Timeline(runtime*1e9)
    tl.show_progress = False

    # set log
    if (log_filename != -1):
        log.set_logger(__name__, tl, log_filename)
        log.set_logger_level('DEBUG')
        log.track_module('COW')
        #log.track_module('timeline')
        log.track_module('light_source')

    qc0 = QuantumChannel("qc0", tl, distance=distance, polarization_fidelity=polarization_fidelity, attenuation=attenuation, phase_noise_coefficient=phase_noise_coefficient, loss=loss)
    qc1 = QuantumChannel("qc1", tl, distance=distance, polarization_fidelity=polarization_fidelity, attenuation=attenuation, phase_noise_coefficient=phase_noise_coefficient, loss=loss)
    cc0 = ClassicalChannel("cc0", tl, distance=distance)
    cc1 = ClassicalChannel("cc1", tl, distance=distance)
    cc0.delay += 1e9  # 1 ms
    cc1.delay += 1e9

    # Alice
    alice = QKDNode("alice", tl, encoding=time_bin_cow, stack_size=1, qkdtype=2, source_type="wcp")
    alice.set_seed(0)

    for name, param in ls_params.items():
        alice.update_lightsource_params(name, param)

    # Bob
    bob = QKDNode("bob", tl, encoding=time_bin_cow, stack_size=1, qkdtype=2, source_type="wcp")
    bob.set_seed(1)

    for i in range(len(detector_params)):
        for name, param in detector_params[i].items():
            bob.update_detector_params(i, name, param)
            
    # ── Set interferometer phase error (Source B) ──
    # Access QSDetectorCOW and set its interferometer phase_error
    from sequence.components.qsdetector_cow import QSDetectorCOW
    for comp in bob.components.values():
        if isinstance(comp, QSDetectorCOW):
            comp.interferometer.phase_error = interferometer_phase_error
            break

    qc0.set_ends(alice, bob.name)
    qc1.set_ends(bob, alice.name)
    cc0.set_ends(alice, bob.name)
    cc1.set_ends(bob, alice.name)

    # COW config
    pair_cow_protocols(alice.protocol_stack[0], bob.protocol_stack[0])
    process = Process(alice.protocol_stack[0], "push", [keysize, key_num, 6e12])
    event = Event(0, process)
    tl.schedule(event)

    tl.init()
    tl.run()

    VISIBILITY = alice.protocol_stack[0].visibility
    QBER, THROUGHPUTS, LATENCY, SKR, LOSS, R_s = _collect_cow_metrics(alice.protocol_stack[0], VISIBILITY, ls_params, distance, attenuation)
    
    return QBER, THROUGHPUTS, LATENCY, SKR, LOSS, R_s, VISIBILITY
    
    
def simulation_BB84_Eve(ls_params, detector_params, runtime=20, log_filename=-1, distance=1e3, polarization_fidelity=0.97, attenuation=0.0002, keysize=256, key_num=math.inf, eve_intercept_rate = 0.9, eve_position = 0.5, source_type="wcp", loss=None):
    tl = Timeline(runtime * 1e9)
    tl.show_progress = False
    if log_filename != -1:
        log.set_logger(__name__, tl, log_filename)
        log.set_logger_level('DEBUG')
        log.track_module('BB84')
        log.track_module('light_source')

    eve = EveNode("eve", tl, intercept_rate=eve_intercept_rate, seed=2)
    qc0 = EveQuantumChannel("qc0", tl, eve_node=eve, distance=distance, polarization_fidelity=polarization_fidelity, attenuation=attenuation, eve_position=eve_position, loss=loss)
    qc1 = QuantumChannel("qc1", tl, distance=distance, polarization_fidelity=polarization_fidelity, attenuation=attenuation, loss=loss)
    
    cc0 = ClassicalChannel("cc0", tl, distance=distance)
    cc1 = ClassicalChannel("cc1", tl, distance=distance)
    cc0.delay += 1e9   # 1 ms
    cc1.delay += 1e9
    
    alice = QKDNode("alice", tl, stack_size=1, source_type=source_type)
    alice.set_seed(0)
    for name, param in ls_params.items():
        alice.update_lightsource_params(name, param)

    bob = QKDNode("bob", tl, stack_size=1, source_type=source_type)
    bob.set_seed(1)
    for i, dp in enumerate(detector_params):
        for name, param in dp.items():
            bob.update_detector_params(i, name, param)

    qc0.set_ends(alice, bob.name)
    qc1.set_ends(bob, alice.name)
    cc0.set_ends(alice, bob.name)
    cc1.set_ends(bob, alice.name)

    
    pair_bb84_protocols(alice.protocol_stack[0], bob.protocol_stack[0])
    tl.schedule(Event(0, Process(alice.protocol_stack[0], "push", [keysize, key_num, 6e12])))
    tl.init()
    tl.run()
    return _collect_metrics(alice.protocol_stack[0], distance, attenuation)
    

def simulation_B92_Eve(ls_params, detector_params, runtime=20, log_filename=-1, distance=1e3, polarization_fidelity=0.97, attenuation=0.0002, keysize=256, key_num=math.inf, eve_intercept_rate = 0.9, eve_position = 0.5, source_type="wcp", loss=None):
    tl = Timeline(runtime * 1e9)
    tl.show_progress = False
    if log_filename != -1:
        log.set_logger(__name__, tl, log_filename)
        log.set_logger_level('DEBUG')
        log.track_module('B92')
        log.track_module('light_source')

    eve = EveNode("eve", tl, intercept_rate=eve_intercept_rate, seed=2)
    qc0 = EveQuantumChannel("qc0", tl, eve_node=eve, distance=distance, polarization_fidelity=polarization_fidelity, attenuation=attenuation, eve_position=eve_position, loss=loss)
    qc1 = QuantumChannel("qc1", tl, distance=distance, polarization_fidelity=polarization_fidelity, attenuation=attenuation, loss=loss)
    
    cc0 = ClassicalChannel("cc0", tl, distance=distance)
    cc1 = ClassicalChannel("cc1", tl, distance=distance)
    cc0.delay += 1e9
    cc1.delay += 1e9

    alice = QKDNode("alice", tl, stack_size=1, qkdtype=1, source_type=source_type)
    alice.set_seed(0)
    for name, param in ls_params.items():
        alice.update_lightsource_params(name, param)

    bob = QKDNode("bob", tl, stack_size=1, qkdtype=1, source_type=source_type)
    bob.set_seed(1)
    for i, dp in enumerate(detector_params):
        for name, param in dp.items():
            bob.update_detector_params(i, name, param)

    qc0.set_ends(alice, bob.name)
    qc1.set_ends(bob, alice.name)
    cc0.set_ends(alice, bob.name)
    cc1.set_ends(bob, alice.name)
 

    pair_b92_protocols(alice.protocol_stack[0], bob.protocol_stack[0])
    tl.schedule(Event(0, Process(alice.protocol_stack[0], "push", [keysize, key_num, 6e12])))
    tl.init()
    tl.run()
    return _collect_metrics(alice.protocol_stack[0], distance, attenuation)

def simulation_COW_Eve(ls_params, detector_params, runtime=20, log_filename=-1, distance=1e3, polarization_fidelity=0.97, attenuation=0.0002, keysize=256, key_num=math.inf, phase_noise_coefficient=0.01, interferometer_phase_error=0.20, eve_intercept_rate = 0.9, eve_position = 0.5, loss=None):
    tl = Timeline(runtime * 1e9)
    tl.show_progress = False
    if log_filename != -1:
        log.set_logger(__name__, tl, log_filename)
        log.set_logger_level('DEBUG')
        log.track_module('COW')
        log.track_module('light_source')

    eve = EveNode("eve", tl, encoding=time_bin_cow, intercept_rate=eve_intercept_rate, seed=2)
    qc0 = EveQuantumChannel("qc0", tl, eve_node=eve, distance=distance, polarization_fidelity=polarization_fidelity, attenuation=attenuation, eve_position=eve_position, phase_noise_coefficient=phase_noise_coefficient, loss=loss)
    qc1 = QuantumChannel("qc1", tl, distance=distance, polarization_fidelity=polarization_fidelity, attenuation=attenuation, phase_noise_coefficient=phase_noise_coefficient, loss=loss)
    
    cc0 = ClassicalChannel("cc0", tl, distance=distance)
    cc1 = ClassicalChannel("cc1", tl, distance=distance)
    cc0.delay += 1e9
    cc1.delay += 1e9

    alice = QKDNode("alice", tl, encoding=time_bin_cow, stack_size=1, qkdtype=2, source_type="wcp")
    alice.set_seed(0)
    for name, param in ls_params.items():
        alice.update_lightsource_params(name, param)

    bob = QKDNode("bob", tl, encoding=time_bin_cow, stack_size=1, qkdtype=2, source_type="wcp")
    bob.set_seed(1)
    for i, dp in enumerate(detector_params):
        for name, param in dp.items():
            bob.update_detector_params(i, name, param)
            
    # ── Set interferometer phase error (Source B) ──
    # Access QSDetectorCOW and set its interferometer phase_error
    from sequence.components.qsdetector_cow import QSDetectorCOW
    for comp in bob.components.values():
        if isinstance(comp, QSDetectorCOW):
            comp.interferometer.phase_error = interferometer_phase_error
            break

    qc0.set_ends(alice, bob.name)
    qc1.set_ends(bob, alice.name)
    cc0.set_ends(alice, bob.name)
    cc1.set_ends(bob, alice.name)

    pair_cow_protocols(alice.protocol_stack[0], bob.protocol_stack[0])
    tl.schedule(Event(0, Process(alice.protocol_stack[0], "push", [keysize, key_num, 6e12])))
    tl.init()
    tl.run()
    
    VISIBILITY = alice.protocol_stack[0].visibility
    QBER, THROUGHPUTS, LATENCY, SKR, LOSS, R_s = _collect_cow_metrics(alice.protocol_stack[0], VISIBILITY, ls_params, distance, attenuation)
    
    return QBER, THROUGHPUTS, LATENCY, SKR, LOSS, R_s, VISIBILITY
    
    
# ═══════════════════════════════════════════════════════════════════════
#  Worker functions for parallel execution
#  (top-level functions required by ProcessPoolExecutor / pickle)
# ═══════════════════════════════════════════════════════════════════════
def _safe_mean(lst, default=np.nan):
    """Return np.nanmean(lst) if lst is non-empty, otherwise default.
    
    Handles None, empty lists, empty numpy arrays, and scalar values
    without triggering RuntimeWarning from numpy.
    """
    if lst is None:
        return default
    try:
        if len(lst) == 0:
            return default
    except TypeError:
        # scalar value (e.g. already a float)
        return float(lst)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = float(np.nanmean(lst))
    return result if not np.isnan(result) else default

def _worker_distance(task: dict):
    """Execute one (protocol, distance) simulation and return a results dict.

    Each worker runs in its own process, so Timeline objects and all
    SeQUeNCe state are completely isolated — no shared-memory conflicts.

    Args:
        task (dict): contains 'protocol', 'distance', and all kwargs
                     needed by the corresponding simulation function.

    Returns:
        dict with keys: protocol, distance, and the simulation outputs.
    """
    proto    = task["protocol"]
    distance = task["distance"]
    kwargs   = task["kwargs"]

    if proto == "BB84":
        QBER, TH, LAT, SKR, LOSS, RS = simulation_BB84(**kwargs)
        return {"protocol": proto, "distance": distance,
                "skr": SKR, "qber": _safe_mean(QBER),
                "throughputs": TH, "latency": LAT, "loss": LOSS, "rs": RS}

    elif proto == "B92":
        QBER, TH, LAT, SKR, LOSS, RS = simulation_B92(**kwargs)
        return {"protocol": proto, "distance": distance,
                "skr": SKR, "qber": _safe_mean(QBER),
                "throughputs": TH, "latency": LAT, "loss": LOSS, "rs": RS}

    elif proto == "COW":
        QBER, TH, LAT, SKR, LOSS, RS, VIS = simulation_COW(**kwargs)
        return {"protocol": proto, "distance": distance,
                "skr": SKR, "qber": _safe_mean(QBER),
                "throughputs": TH, "latency": LAT, "loss": LOSS, "rs": RS,
                "visibility": _safe_mean(VIS)}

    elif proto == "BB84+Eve":
        QBER, TH, LAT, SKR, LOSS, RS = simulation_BB84_Eve(**kwargs)
        return {"protocol": proto, "distance": distance,
                "skr": SKR, "qber": _safe_mean(QBER),
                "throughputs": TH, "latency": LAT, "loss": LOSS, "rs": RS}

    elif proto == "B92+Eve":
        QBER, TH, LAT, SKR, LOSS, RS = simulation_B92_Eve(**kwargs)
        return {"protocol": proto, "distance": distance,
                "skr": SKR, "qber": _safe_mean(QBER),
                "throughputs": TH, "latency": LAT, "loss": LOSS, "rs": RS}

    elif proto == "COW+Eve":
        QBER, TH, LAT, SKR, LOSS, RS, VIS = simulation_COW_Eve(**kwargs)
        return {"protocol": proto, "distance": distance,
                "skr": SKR, "qber": _safe_mean(QBER),
                "throughputs": TH, "latency": LAT, "loss": LOSS, "rs": RS,
                "visibility": _safe_mean(VIS)}

    else:
        raise ValueError(f"Unknown protocol: {proto}")


def _worker_keysize(task: dict):
    """Execute one (protocol, keysize) simulation and return a results dict.

    Identical logic to _worker_distance but keyed on 'keysize' instead.
    """
    proto   = task["protocol"]
    keysize = task["keysize"]
    kwargs  = task["kwargs"]

    if proto == "BB84":
        QBER, TH, LAT, SKR, LOSS, RS = simulation_BB84(**kwargs)
        return {"protocol": proto, "keysize": keysize,
                "skr": SKR, "qber": _safe_mean(QBER),
                "throughputs": TH, "latency": LAT, "loss": LOSS, "rs": RS}

    elif proto == "B92":
        QBER, TH, LAT, SKR, LOSS, RS = simulation_B92(**kwargs)
        return {"protocol": proto, "keysize": keysize,
                "skr": SKR, "qber": _safe_mean(QBER),
                "throughputs": TH, "latency": LAT, "loss": LOSS, "rs": RS}

    elif proto == "COW":
        QBER, TH, LAT, SKR, LOSS, RS, VIS = simulation_COW(**kwargs)
        return {"protocol": proto, "keysize": keysize,
                "skr": SKR, "qber": _safe_mean(QBER),
                "throughputs": TH, "latency": LAT, "loss": LOSS, "rs": RS,
                "visibility": _safe_mean(VIS)}

    elif proto == "BB84+Eve":
        QBER, TH, LAT, SKR, LOSS, RS = simulation_BB84_Eve(**kwargs)
        return {"protocol": proto, "keysize": keysize,
                "skr": SKR, "qber": _safe_mean(QBER),
                "throughputs": TH, "latency": LAT, "loss": LOSS, "rs": RS}

    elif proto == "B92+Eve":
        QBER, TH, LAT, SKR, LOSS, RS = simulation_B92_Eve(**kwargs)
        return {"protocol": proto, "keysize": keysize,
                "skr": SKR, "qber": _safe_mean(QBER),
                "throughputs": TH, "latency": LAT, "loss": LOSS, "rs": RS}

    elif proto == "COW+Eve":
        QBER, TH, LAT, SKR, LOSS, RS, VIS = simulation_COW_Eve(**kwargs)
        return {"protocol": proto, "keysize": keysize,
                "skr": SKR, "qber": _safe_mean(QBER),
                "throughputs": TH, "latency": LAT, "loss": LOSS, "rs": RS,
                "visibility": _safe_mean(VIS)}

    else:
        raise ValueError(f"Unknown protocol: {proto}")



# ═══════════════════════════════════════════════════════════════════════
#  Parallel sweep functions
# ═══════════════════════════════════════════════════════════════════════

def _build_distance_tasks(runtime, d_list, channel_parameters, ls_params_cow, ls_params,
                          detector_params, detector_params_cow, keysize, key_num, loss_parameters):
    """Build a flat list of task dicts for every (protocol × distance) pair."""
    att  = channel_parameters[1]
    pfid = channel_parameters[2]
    tasks = []

    for d in d_list:
        common = dict(runtime=runtime, distance=d,
                      polarization_fidelity=pfid, attenuation=att, keysize=keysize, key_num=key_num)
        loss = channel_FSO_loss(distance=d, wavelength=ls_params["wavelength"], v_range=loss_parameters["v_range"],
                     receiver_radius=loss_parameters["receiver_radius"], pressure=loss_parameters["pressure"], temperature=loss_parameters["temperature"], w_0=loss_parameters["w_0"], C_T=loss_parameters["C_T"], R_0=loss_parameters["R_0"], friction_velocity=loss_parameters["friction_velocity"], height=loss_parameters["height"],
                     size_raindrop=loss_parameters["size_raindrop"], viscosity=loss_parameters["viscosity"], precipitation_rate=loss_parameters["precipitation_rate"], Q_scat=loss_parameters["Q_scat"])

        tasks.append({"protocol": "BB84", "distance": d,
                      "kwargs": {**common, "ls_params": ls_params,
                                 "detector_params": detector_params,
                                 "source_type": "sps",
                                 "loss": loss}})

        tasks.append({"protocol": "B92", "distance": d,
                      "kwargs": {**common, "ls_params": ls_params,
                                 "detector_params": detector_params,
                                 "source_type": "sps",
                                 "loss": loss}})

        tasks.append({"protocol": "COW", "distance": d,
                      "kwargs": {**common, "ls_params": ls_params_cow,
                                 "detector_params": detector_params_cow,
                                 "loss": loss}})

        tasks.append({"protocol": "BB84+Eve", "distance": d,
                      "kwargs": {**common, "ls_params": ls_params,
                                 "detector_params": detector_params,
                                 "source_type": "sps",
                                 "loss": loss}})

        tasks.append({"protocol": "B92+Eve", "distance": d,
                      "kwargs": {**common, "ls_params": ls_params,
                                 "detector_params": detector_params,
                                 "source_type": "sps",
                                 "loss": loss}})

        tasks.append({"protocol": "COW+Eve", "distance": d,
                      "kwargs": {**common, "ls_params": ls_params_cow,
                                 "detector_params": detector_params_cow,
                                 "loss": loss}})
    return tasks


def _build_keysize_tasks(runtime, keysize_list, channel_parameters, ls_params_cow, ls_params,
                         detector_params, detector_params_cow, key_num, loss_parameters):
    """Build a flat list of task dicts for every (protocol × keysize) pair."""
    dist = channel_parameters[0]
    att  = channel_parameters[1]
    pfid = channel_parameters[2]
    loss = channel_FSO_loss(distance=dist, wavelength=ls_params["wavelength"], v_range=loss_parameters["v_range"],
                     receiver_radius=loss_parameters["receiver_radius"], pressure=loss_parameters["pressure"], temperature=loss_parameters["temperature"], w_0=loss_parameters["w_0"], C_T=loss_parameters["C_T"], R_0=loss_parameters["R_0"], friction_velocity=loss_parameters["friction_velocity"], height=loss_parameters["height"],
                     size_raindrop=loss_parameters["size_raindrop"], viscosity=loss_parameters["viscosity"], precipitation_rate=loss_parameters["precipitation_rate"], Q_scat=loss_parameters["Q_scat"])
    tasks = []

    for k in keysize_list:
        common = dict(runtime=runtime, distance=dist,
                      polarization_fidelity=pfid, attenuation=att, keysize=k, key_num=key_num)

        tasks.append({"protocol": "BB84", "keysize": k,
                      "kwargs": {**common, "ls_params": ls_params,
                                 "detector_params": detector_params,
                                 "source_type": "sps",
                                 "loss": loss}})

        tasks.append({"protocol": "B92", "keysize": k,
                      "kwargs": {**common, "ls_params": ls_params,
                                 "detector_params": detector_params,
                                 "source_type": "sps",
                                 "loss": loss}})

        tasks.append({"protocol": "COW", "keysize": k,
                      "kwargs": {**common, "ls_params": ls_params_cow,
                                 "detector_params": detector_params_cow,
                                 "loss": loss}})

        tasks.append({"protocol": "BB84+Eve", "keysize": k,
                      "kwargs": {**common, "ls_params": ls_params,
                                 "detector_params": detector_params,
                                 "source_type": "sps",
                                 "loss": loss}})

        tasks.append({"protocol": "B92+Eve", "keysize": k,
                      "kwargs": {**common, "ls_params": ls_params,
                                 "detector_params": detector_params,
                                 "source_type": "sps",
                                 "loss": loss}})

        tasks.append({"protocol": "COW+Eve", "keysize": k,
                      "kwargs": {**common, "ls_params": ls_params_cow,
                                 "detector_params": detector_params_cow,
                                 "loss": loss}})
    return tasks


def _collect_distance_results(d_list, results_list):
    """Organise raw worker results into the metrics dict keyed by distance."""
    # Pre-allocate per-protocol containers
    proto_keys = ["BB84", "B92", "COW", "BB84+Eve", "B92+Eve", "COW+Eve"]
    data = {p: {} for p in proto_keys}     # protocol -> {distance: result_dict}

    for r in results_list:
        data[r["protocol"]][r["distance"]] = r

    # Build ordered metric lists
    metrics = {"distance": np.array(d_list)}
    for suffix, proto in [("BB84", "BB84"), ("B92", "B92"), ("COW", "COW"),
                          ("BB84+Eve", "BB84+Eve"), ("B92+Eve", "B92+Eve"),
                          ("COW+Eve", "COW+Eve")]:
        skr, qber, th, lat, loss, rs = [], [], [], [], [], []
        vis = []
        for d in d_list:
            r = data[proto].get(d, {})
            skr.append(r.get("skr", np.nan))
            qber.append(r.get("qber", np.nan))
            th.append(r.get("throughputs", np.nan))
            lat.append(r.get("latency", np.nan))
            loss.append(r.get("loss", np.nan))
            rs.append(r.get("rs", np.nan))
            if "COW" in proto:
                vis.append(r.get("visibility", np.nan))

        metrics[f"R_sk-{suffix}"]       = np.array(skr)
        metrics[f"QBER-{suffix}"]       = np.array(qber)
        metrics[f"Throughputs-{suffix}"] = np.array(th)
        metrics[f"Latency-{suffix}"]    = np.array(lat)
        metrics[f"Loss-{suffix}"]       = np.array(loss)
        metrics[f"R_s-{suffix}"]        = np.array(rs)
        if "COW" in suffix:
            metrics[f"Visibility-{suffix}"] = np.array(vis)

    return metrics


def _collect_keysize_results(keysize_list, results_list):
    """Organise raw worker results into the metrics dict keyed by keysize."""
    proto_keys = ["BB84", "B92", "COW", "BB84+Eve", "B92+Eve", "COW+Eve"]
    data = {p: {} for p in proto_keys}

    for r in results_list:
        data[r["protocol"]][r["keysize"]] = r

    metrics = {"keysize": np.array(keysize_list)}
    for suffix, proto in [("BB84", "BB84"), ("B92", "B92"), ("COW", "COW"),
                          ("BB84+Eve", "BB84+Eve"), ("B92+Eve", "B92+Eve"),
                          ("COW+Eve", "COW+Eve")]:
        skr, qber, th, lat, loss, rs = [], [], [], [], [], []
        vis = []
        for k in keysize_list:
            r = data[proto].get(k, {})
            skr.append(r.get("skr", np.nan))
            qber.append(r.get("qber", np.nan))
            th.append(r.get("throughputs", np.nan))
            lat.append(r.get("latency", np.nan))
            loss.append(r.get("loss", np.nan))
            rs.append(r.get("rs", np.nan))
            if "COW" in proto:
                vis.append(r.get("visibility", np.nan))

        metrics[f"R_sk-{suffix}"]       = np.array(skr)
        metrics[f"QBER-{suffix}"]       = np.array(qber)
        metrics[f"Throughputs-{suffix}"] = np.array(th)
        metrics[f"Latency-{suffix}"]    = np.array(lat)
        metrics[f"Loss-{suffix}"]       = np.array(loss)
        metrics[f"R_s-{suffix}"]        = np.array(rs)
        if "COW" in suffix:
            metrics[f"Visibility-{suffix}"] = np.array(vis)

    return metrics


# ═══════════════════════════════════════════════════════════════════════
#  Public parallel entry-points (drop-in replacements)
# ═══════════════════════════════════════════════════════════════════════

def sim_variable_distance(runtime, d_step, d_lim, channel_parameters,
                          ls_params_cow, ls_params, detector_params, detector_params_cow,
                          keysize, key_num, loss_parameters, max_workers=None):
    """Parallel version of the original sim_variable_distance.

    Args:
        max_workers (int | None): number of parallel processes.
            Defaults to the number of CPU cores.
    """
    if max_workers is None:
        max_workers = os.cpu_count() or 4

    d_list = list(range(d_step, d_lim + 1, d_step))
        
    tasks = _build_distance_tasks(
        runtime, d_list, channel_parameters,
        ls_params_cow, ls_params, detector_params, detector_params_cow, keysize, key_num, loss_parameters)

    total = len(tasks)
    results_list = []

    print(f"[parallel] Launching {total} tasks across {max_workers} workers "
          f"({len(d_list)} distances × 6 protocols)")

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_worker_distance, t): t for t in tasks}
        for i, future in enumerate(as_completed(futures), 1):
            task_info = futures[future]
            try:
                result = future.result()
                results_list.append(result)
            except Exception as exc:
                proto = task_info["protocol"]
                dist  = task_info["distance"]
                print(f"\n[parallel] WARNING: {proto} @ d={dist}m failed: {exc}")
                # Insert a NaN placeholder so the rest of the simulation continues
                fallback = {"protocol": proto, "distance": dist,
                            "skr": np.nan, "qber": np.nan, "throughputs": np.nan,
                            "latency": np.nan, "loss": np.nan, "rs": np.nan}
                if "COW" in proto:
                    fallback["visibility"] = np.nan
                results_list.append(fallback)
            pct = i / total * 100
            print(f"\r[parallel] {i}/{total} done ({pct:.1f}%)", end="", flush=True)

    print()  # newline after progress

    metrics = _collect_distance_results(d_list, results_list)
        
    pd.DataFrame(metrics).to_csv('data/metrics_variable-distance.csv', index=False)
    print("[parallel] Saved metrics_variable-distance.csv")

def sim_variable_keysize(runtime, keysize_list, channel_parameters,
                         ls_params_cow, ls_params, detector_params, detector_params_cow, key_num,
                         loss_parameters, max_workers=None):
    """Parallel version of the original sim_variable_keysize."""
    if max_workers is None:
        max_workers = os.cpu_count() or 4

    tasks = _build_keysize_tasks(
        runtime, keysize_list, channel_parameters,
        ls_params_cow, ls_params, detector_params, detector_params_cow, key_num, loss_parameters)

    total = len(tasks)
    results_list = []

    print(f"[parallel] Launching {total} tasks across {max_workers} workers "
          f"({len(keysize_list)} keysizes × 6 protocols)")

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_worker_keysize, t): t for t in tasks}
        for i, future in enumerate(as_completed(futures), 1):
            task_info = futures[future]
            try:
                result = future.result()
                results_list.append(result)
            except Exception as exc:
                proto = task_info["protocol"]
                ks    = task_info["keysize"]
                print(f"\n[parallel] WARNING: {proto} @ keysize={ks} failed: {exc}")
                fallback = {"protocol": proto, "keysize": ks,
                            "skr": np.nan, "qber": np.nan, "throughputs": np.nan,
                            "latency": np.nan, "loss": np.nan, "rs": np.nan}
                if "COW" in proto:
                    fallback["visibility"] = np.nan
                results_list.append(fallback)

            pct = i / total * 100
            print(f"\r[parallel] {i}/{total} done ({pct:.1f}%)", end="", flush=True)

    print()

    metrics = _collect_keysize_results(keysize_list, results_list)

    pd.DataFrame(metrics).to_csv('data/metrics_variable-keysize.csv', index=False)
    print("[parallel] Saved metrics_variable-keysize.csv")

def run_simulation():
    start = time.time()

    ls_params = {"frequency": 8e6, "wavelength":780, "mean_photon_num": 1}
    ls_params_cow = {"frequency": 8e6, "wavelength":780, "mean_photon_num": 0.5}
    detector_params = [{"efficiency": 0.65, "dark_count": 100, "time_resolution": 1000, "count_rate": 20e6},
                       {"efficiency": 0.65, "dark_count": 100, "time_resolution": 1000, "count_rate": 20e6}]
    detector_params_cow = [{"efficiency": 0.65, "dark_count": 100, "time_resolution": 1000, "count_rate": 20e6},
                       {"efficiency": 0.65, "dark_count": 100, "time_resolution": 1000, "count_rate": 20e6},
                       {"efficiency": 0.65, "dark_count": 100, "time_resolution": 1000, "count_rate": 20e6}]
    keysize = 10000
    key_num = 1
    # channel_parameters = (distance [in meters], attenuation [in dB/m], polarization_fidelity [in %])
    channel_parameters = (700, 0.0002, 0.97)
    
    # Source of information on the factors that influence signal loss:
    # From transmitter:
    #     w_0:
    #     R_0: para feixes colimados, adota-se R_0 = math.inf
    #     transmitter_height:
    # From receiver:
    #     receiver_radius:
    #     receiver_height:
    # From channel:
    #     v_range:
    #     pressure: https://www.labmicro.iag.usp.br/Data/data_PMIAG.html
    #     temperature: https://www.labmicro.iag.usp.br/Data/data_PMIAG.html
    #     C_T:
    #     size_raindrop:
    #     viscosity: Sutherland, W. (1893), "The viscosity of gases and molecular force", Philosophical Magazine, S. 5, 36, pp. 507-531 (1893).
    #     precipitation_rate: https://www.labmicro.iag.usp.br/Data/data_PMIAG.html
    #     Q_scat: Calculado utilizando 'Prahl, S. (2026). miepython: Pure python calculation of Mie scattering (Version 3.2.0) 
    #               [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.7949263' e 'Xiaohong Quan and Edward S. Fry, "Empirical equation for the index of 
    #               refraction of seawater," Appl. Opt. 34, 3477-3480 (1995)'
    #     friction_velocity: https://www.labmicro.iag.usp.br/Data/data_PMIAG.html
    #     height = (transmitter_height+receiver_height)/2
    loss_parameters = {"v_range":,
                       "receiver_radius":, "pressure":, "temperature":, "w_0":, "C_T":, "R_0":math.inf, "friction_velocity":, "height":,
                       "size_raindrop":, "viscosity":None, "precipitation_rate":, "Q_scat":2}
    sim_variable_distance(runtime=1000, d_step=1000, d_lim=100000, channel_parameters=channel_parameters, ls_params_cow=ls_params_cow, ls_params=ls_params, detector_params=detector_params, detector_params_cow=detector_params_cow, keysize=keysize, key_num=key_num, loss_parameters=loss_parameters)
    sim_variable_keysize(runtime=1000, keysize_list=[20, 45, 50, 100, 200, 400, 800, 1600, 5000, 20000, 40000, 80000, 100000], channel_parameters=channel_parameters, ls_params_cow=ls_params_cow, ls_params=ls_params, detector_params=detector_params, detector_params_cow=detector_params_cow, key_num=key_num, loss_parameters=loss_parameters)

    end = time.time()
    
    simulator_metrics = {"Total_execution_time_(seconds)": [end-start]}
    pd.DataFrame(simulator_metrics).to_csv('data/simulator_metrics.csv', index=False)

if __name__ == "__main__":
    run_simulation()
