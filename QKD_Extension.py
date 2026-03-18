import math

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
import sequence.utils.log as log
import numpy as np
import pandas as pd

def safe_log10(lst: list) -> np.ndarray:
    arr = np.array(lst, dtype=float)
    arr[arr <= 0] = np.nan
    return np.log10(arr)


def binary_entropy(Q):
    if Q == 0 or Q == 1:
        return 0
    return -Q * math.log2(Q) - (1 - Q) * math.log2(1 - Q)
    
def _collect_metrics(protocol, distance: float, attenuation: float):
    """Extrai QBER, throughput, latência, SKR e perda de um protocolo."""
    secret_key_rate_mean = 0

    QBER = protocol.error_rates
    THROUGHPUTS = np.mean(protocol.throughputs)
    LATENCY = protocol.latency
    R_s_list = []
    for i, e in enumerate(QBER):
        R_s = protocol.sifted_bits_length[i] / protocol.send_bits_length
        secret_key_rate_mean += R_s * (1 - binary_entropy(e))
        R_s_list.append(R_s)
    SECRET_KEY_RATE = secret_key_rate_mean/len(QBER)
    mean_R_s = np.mean(np.array(R_s_list, dtype=float))
    LOSS = 1-10**((distance*attenuation)/(-10))
    return QBER, THROUGHPUTS, LATENCY, SECRET_KEY_RATE, LOSS, mean_R_s
    
def _collect_cow_metrics(protocol, visibility, ls_params, distance: float, attenuation: float):
    """Extrai métricas COW com SKR ajustado por visibilidade do monitoramento."""
    secret_key_rate_mean = 0

    qber_list = protocol.error_rates
    throughputs = np.mean(protocol.throughputs)
    latency = protocol.latency
    
    # R_sk calculated based on https://doi.org/10.1063/1.2126792
    t = 10 ** ((distance * attenuation) / (-10))
    loss = 1 - t
    
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



def simulation_BB84(ls_params, detector_params, runtime=20, log_filename=-1, distance=1e3, polarization_fidelity=0.97, attenuation=0.0002, keysize=256, key_num=math.inf, source_type="wcp"):
    tl = Timeline(runtime*1e9)
    tl.show_progress = True

    # set log
    if (log_filename != -1):
        log.set_logger(__name__, tl, log_filename)
        log.set_logger_level('DEBUG')
        log.track_module('BB84')
        #log.track_module('timeline')
        log.track_module('light_source')

    qc0 = QuantumChannel("qc0", tl, distance=distance, polarization_fidelity=polarization_fidelity, attenuation=attenuation)
    qc1 = QuantumChannel("qc1", tl, distance=distance, polarization_fidelity=polarization_fidelity, attenuation=attenuation)
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

def simulation_B92(ls_params, detector_params, runtime=20, log_filename=-1, distance=1e3, polarization_fidelity=0.97, attenuation=0.0002, keysize=256, key_num=math.inf, source_type="wcp"):
    tl = Timeline(runtime*1e9)
    tl.show_progress = True

    # set log
    if (log_filename != -1):
        log.set_logger(__name__, tl, log_filename)
        log.set_logger_level('DEBUG')
        log.track_module('B92')
        #log.track_module('timeline')
        log.track_module('light_source')

    qc0 = QuantumChannel("qc0", tl, distance=distance, polarization_fidelity=polarization_fidelity, attenuation=attenuation)
    qc1 = QuantumChannel("qc1", tl, distance=distance, polarization_fidelity=polarization_fidelity, attenuation=attenuation)
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


def simulation_COW(ls_params, detector_params, runtime=20, log_filename=-1, distance=1e3, polarization_fidelity=0.97, attenuation=0.0002, keysize=256, key_num=math.inf):
    tl = Timeline(runtime*1e9)
    tl.show_progress = True

    # set log
    if (log_filename != -1):
        log.set_logger(__name__, tl, log_filename)
        log.set_logger_level('DEBUG')
        log.track_module('COW')
        #log.track_module('timeline')
        log.track_module('light_source')

    qc0 = QuantumChannel("qc0", tl, distance=distance, polarization_fidelity=polarization_fidelity, attenuation=attenuation)
    qc1 = QuantumChannel("qc1", tl, distance=distance, polarization_fidelity=polarization_fidelity, attenuation=attenuation)
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
    
    
def simulation_BB84_Eve(ls_params, detector_params, runtime=20, log_filename=-1, distance=1e3, polarization_fidelity=0.97, attenuation=0.0002, keysize=256, key_num=math.inf, eve_intercept_rate = 1.0, eve_position = 0.5, source_type="wcp"):
    dist_ae = distance * eve_position
    dist_eb = distance * (1.0 - eve_position)

    tl = Timeline(runtime * 1e9)
    tl.show_progress = True
    if log_filename != -1:
        log.set_logger(__name__, tl, log_filename)
        log.set_logger_level('DEBUG')
        log.track_module('BB84')
        log.track_module('light_source')

    eve = EveNode("eve", tl, intercept_rate=eve_intercept_rate, seed=2)
    qc0 = EveQuantumChannel("qc0", tl, eve_node=eve, distance=distance, polarization_fidelity=polarization_fidelity, attenuation=attenuation, eve_position=eve_position)
    qc1 = QuantumChannel("qc1", tl, distance=distance, polarization_fidelity=polarization_fidelity, attenuation=attenuation)
    
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
    

def simulation_B92_Eve(ls_params, detector_params, runtime=20, log_filename=-1, distance=1e3, polarization_fidelity=0.97, attenuation=0.0002, keysize=256, key_num=math.inf, eve_intercept_rate = 1.0, eve_position = 0.5, source_type="wcp"):
    dist_ae = distance * eve_position
    dist_eb = distance * (1.0 - eve_position)

    tl = Timeline(runtime * 1e9)
    tl.show_progress = True
    if log_filename != -1:
        log.set_logger(__name__, tl, log_filename)
        log.set_logger_level('DEBUG')
        log.track_module('B92')
        log.track_module('light_source')

    eve = EveNode("eve", tl, intercept_rate=eve_intercept_rate, seed=2)
    qc0 = EveQuantumChannel("qc0", tl, eve_node=eve, distance=distance, polarization_fidelity=polarization_fidelity, attenuation=attenuation, eve_position=eve_position)
    qc1 = QuantumChannel("qc1", tl, distance=distance, polarization_fidelity=polarization_fidelity, attenuation=attenuation)
    
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

def simulation_COW_Eve(ls_params, detector_params, runtime=20, log_filename=-1, distance=1e3, polarization_fidelity=0.97, attenuation=0.0002, keysize=256, key_num=math.inf, eve_intercept_rate = 1.0, eve_position = 0.5):
    dist_ae = distance * eve_position
    dist_eb = distance * (1.0 - eve_position)

    tl = Timeline(runtime * 1e9)
    tl.show_progress = True
    if log_filename != -1:
        log.set_logger(__name__, tl, log_filename)
        log.set_logger_level('DEBUG')
        log.track_module('COW')
        log.track_module('light_source')

    eve = EveNode("eve", tl, encoding=time_bin_cow, intercept_rate=eve_intercept_rate, seed=2)
    qc0 = EveQuantumChannel("qc0", tl, eve_node=eve, distance=distance, polarization_fidelity=polarization_fidelity, attenuation=attenuation, eve_position=eve_position)
    qc1 = QuantumChannel("qc1", tl, distance=distance, polarization_fidelity=polarization_fidelity, attenuation=attenuation)
    
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
    
    
def sim_variable_distance(d_step, d_lim, channel_parameters, ls_params, detector_params, detector_params_cow, keysize):
    skr_bb84, qber_bb84, throughputs_bb84, latency_bb84, loss_bb84, rs_bb84 = [], [], [], [], [], []
    skr_b92, qber_b92, throughputs_b92, latency_b92, loss_b92, rs_b92 = [], [], [], [], [], []
    skr_cow, qber_cow, throughputs_cow, latency_cow, loss_cow, rs_cow, visibility_cow = [], [], [], [], [], [], []
    
    skr_bb84e, qber_bb84e, throughputs_bb84e, latency_bb84e, loss_bb84e, rs_bb84e = [], [], [], [], [], []
    skr_b92e, qber_b92e, throughputs_b92e, latency_b92e, loss_b92e, rs_b92e = [], [], [], [], [], []
    skr_cowe, qber_cowe, throughputs_cowe, latency_cowe, loss_cowe, rs_cowe, visibility_cowe = [], [], [], [], [], [], []
    d_list = []
    d = 0
    while d <= d_lim:
        # Sem Eve (Cenário Ideal)
        QBER_BB84, THROUGHPUTS_BB84, LATENCY_BB84, SECRET_KEY_RATE_BB84, LOSS_BB84, RS_BB84 = simulation_BB84(ls_params, detector_params, distance=d, polarization_fidelity=channel_parameters[2], attenuation=channel_parameters[1], keysize=keysize, source_type="sps")
        QBER_B92, THROUGHPUTS_B92, LATENCY_B92, SECRET_KEY_RATE_B92, LOSS_B92, RS_B92 = simulation_B92(ls_params, detector_params, distance=d, polarization_fidelity=channel_parameters[2], attenuation=channel_parameters[1], keysize=keysize, source_type="sps")
        QBER_COW, THROUGHPUTS_COW, LATENCY_COW, SECRET_KEY_RATE_COW, LOSS_COW, RS_COW, VISIBILITY_COW = simulation_COW(ls_params, detector_params_cow, distance=d, polarization_fidelity=channel_parameters[2], attenuation=channel_parameters[1], keysize=keysize)

        # Com Eve
        QBER_BB84e, THROUGHPUTS_BB84e, LATENCY_BB84e, SECRET_KEY_RATE_BB84e, LOSS_BB84e, RS_BB84e = simulation_BB84_Eve(ls_params, detector_params, distance=d, polarization_fidelity=channel_parameters[2], attenuation=channel_parameters[1], keysize=keysize, source_type="sps")
        QBER_B92e, THROUGHPUTS_B92e, LATENCY_B92e, SECRET_KEY_RATE_B92e, LOSS_B92e, RS_B92e = simulation_B92_Eve(ls_params, detector_params, distance=d, polarization_fidelity=channel_parameters[2], attenuation=channel_parameters[1], keysize=keysize, source_type="sps")
        QBER_COWe, THROUGHPUTS_COWe, LATENCY_COWe, SECRET_KEY_RATE_COWe, LOSS_COWe, RS_COWe, VISIBILITY_COWe = simulation_COW_Eve(ls_params, detector_params_cow, distance=d, polarization_fidelity=channel_parameters[2], attenuation=channel_parameters[1], keysize=keysize)
        
        d_list.append(d)
        
        skr_bb84.append(SECRET_KEY_RATE_BB84); qber_bb84.append(np.mean(QBER_BB84)); throughputs_bb84.append(THROUGHPUTS_BB84); latency_bb84.append(LATENCY_BB84); loss_bb84.append(LOSS_BB84); rs_bb84.append(RS_BB84)
        skr_b92.append(SECRET_KEY_RATE_B92); qber_b92.append(np.mean(QBER_B92)); throughputs_b92.append(THROUGHPUTS_B92); latency_b92.append(LATENCY_B92); loss_b92.append(LOSS_B92); rs_b92.append(RS_B92)
        skr_cow.append(SECRET_KEY_RATE_COW); qber_cow.append(np.mean(QBER_COW)); throughputs_cow.append(THROUGHPUTS_COW); latency_cow.append(LATENCY_COW); loss_cow.append(LOSS_COW); rs_cow.append(RS_COW); visibility_cow.append(np.mean(VISIBILITY_COW))
        
        skr_bb84e.append(SECRET_KEY_RATE_BB84e); qber_bb84e.append(np.mean(QBER_BB84e)); throughputs_bb84e.append(THROUGHPUTS_BB84e); latency_bb84e.append(LATENCY_BB84e); loss_bb84e.append(LOSS_BB84e); rs_bb84e.append(RS_BB84e)
        skr_b92e.append(SECRET_KEY_RATE_B92e); qber_b92e.append(np.mean(QBER_B92e)); throughputs_b92e.append(THROUGHPUTS_B92e); latency_b92e.append(LATENCY_B92e); loss_b92e.append(LOSS_B92e); rs_b92e.append(RS_B92e)
        skr_cowe.append(SECRET_KEY_RATE_COWe); qber_cowe.append(np.mean(QBER_COWe)); throughputs_cowe.append(THROUGHPUTS_COWe); latency_cowe.append(LATENCY_COWe); loss_cowe.append(LOSS_COWe); rs_cowe.append(RS_COWe); visibility_cowe.append(np.mean(VISIBILITY_COWe))
        
        print()
        print("Simulation "+str((d/d_lim)*100)+'% completed')
        d += d_step
        
    metrics = {
        "distance":        np.array(d_list),
        "R_sk-BB84":       np.array(skr_bb84),
        "QBER-BB84":       qber_bb84,
        "Throughputs-BB84": np.array(throughputs_bb84),
        "Latency-BB84": np.array(latency_bb84),
        "Loss-BB84": np.array(loss_bb84),
        "R_s-BB84": np.array(rs_bb84),
        "R_sk-B92":        np.array(skr_b92),
        "QBER-B92":        qber_b92,
        "Throughputs-B92": np.array(throughputs_b92),
        "Latency-B92": np.array(latency_b92),
        "Loss-B92": np.array(loss_b92),
        "R_s-B92": np.array(rs_b92),
        "R_sk-COW":        np.array(skr_cow),
        "QBER-COW":        qber_cow,
        "Throughputs-COW": np.array(throughputs_cow),
        "Latency-COW": np.array(latency_cow),
        "Loss-COW": np.array(loss_cow),
        "R_s-COW": np.array(rs_cow),
        "Visibility-COW": np.array(visibility_cow),
        "R_sk-BB84+Eve":   np.array(skr_bb84e),
        "QBER-BB84+Eve":   qber_bb84e,
        "Throughputs-BB84+Eve": np.array(throughputs_bb84e),
        "Latency-BB84+Eve": np.array(latency_bb84e),
        "Loss-BB84+Eve": np.array(loss_bb84e),
        "R_s-BB84+Eve": np.array(rs_bb84e),
        "R_sk-B92+Eve":    np.array(skr_b92e),
        "QBER-B92+Eve":    qber_b92e,
        "Throughputs-B92+Eve": np.array(throughputs_b92e),
        "Latency-B92+Eve": np.array(latency_b92e),
        "Loss-B92+Eve": np.array(loss_b92e),
        "R_s-B92+Eve": np.array(rs_b92e),
        "R_sk-COW+Eve":    np.array(skr_cowe),
        "QBER-COW+Eve":    qber_cowe,
        "Throughputs-COW+Eve": np.array(throughputs_cowe),
        "Latency-COW+Eve": np.array(latency_cowe),
        "Loss-COW+Eve": np.array(loss_cowe),
        "R_s-COW+Eve": np.array(rs_cowe),
        "Visibility-COW+Eve": np.array(visibility_cowe)
    }
    pd.DataFrame(metrics).to_csv('metrics_variable-distance.csv', index=False)

def sim_variable_keysize(keysize_list, channel_parameters, ls_params, detector_params, detector_params_cow):
    skr_bb84, qber_bb84, throughputs_bb84, latency_bb84, loss_bb84, rs_bb84 = [], [], [], [], [], []
    skr_b92, qber_b92, throughputs_b92, latency_b92, loss_b92, rs_b92 = [], [], [], [], [], []
    skr_cow, qber_cow, throughputs_cow, latency_cow, loss_cow, rs_cow, visibility_cow = [], [], [], [], [], [], []
    
    skr_bb84e, qber_bb84e, throughputs_bb84e, latency_bb84e, loss_bb84e, rs_bb84e = [], [], [], [], [], []
    skr_b92e, qber_b92e, throughputs_b92e, latency_b92e, loss_b92e, rs_b92e = [], [], [], [], [], []
    skr_cowe, qber_cowe, throughputs_cowe, latency_cowe, loss_cowe, rs_cowe, visibility_cowe = [], [], [], [], [], [], []

    for k in keysize_list:
        # Sem Eve (Cenário Ideal)
        QBER_BB84, THROUGHPUTS_BB84, LATENCY_BB84, SECRET_KEY_RATE_BB84, LOSS_BB84, RS_BB84 = simulation_BB84(ls_params, detector_params, distance=channel_parameters[0], polarization_fidelity=channel_parameters[2], attenuation=channel_parameters[1], keysize=k, source_type="sps")
        QBER_B92, THROUGHPUTS_B92, LATENCY_B92, SECRET_KEY_RATE_B92, LOSS_B92, RS_B92 = simulation_B92(ls_params, detector_params, distance=channel_parameters[0], polarization_fidelity=channel_parameters[2], attenuation=channel_parameters[1], keysize=k, source_type="sps")
        QBER_COW, THROUGHPUTS_COW, LATENCY_COW, SECRET_KEY_RATE_COW, LOSS_COW, RS_COW, VISIBILITY_COW = simulation_COW(ls_params, detector_params_cow, distance=channel_parameters[0], polarization_fidelity=channel_parameters[2], attenuation=channel_parameters[1], keysize=k)

        # Com Eve
        QBER_BB84e, THROUGHPUTS_BB84e, LATENCY_BB84e, SECRET_KEY_RATE_BB84e, LOSS_BB84e, RS_BB84e = simulation_BB84_Eve(ls_params, detector_params, distance=channel_parameters[0], polarization_fidelity=channel_parameters[2], attenuation=channel_parameters[1], keysize=k, source_type="sps")
        QBER_B92e, THROUGHPUTS_B92e, LATENCY_B92e, SECRET_KEY_RATE_B92e, LOSS_B92e, RS_B92e = simulation_B92_Eve(ls_params, detector_params, distance=channel_parameters[0], polarization_fidelity=channel_parameters[2], attenuation=channel_parameters[1], keysize=k, source_type="sps")
        QBER_COWe, THROUGHPUTS_COWe, LATENCY_COWe, SECRET_KEY_RATE_COWe, LOSS_COWe, RS_COWe, VISIBILITY_COWe = simulation_COW_Eve(ls_params, detector_params_cow, distance=channel_parameters[0], polarization_fidelity=channel_parameters[2], attenuation=channel_parameters[1], keysize=k)
        
        d_list.append(d)
        
        skr_bb84.append(SECRET_KEY_RATE_BB84); qber_bb84.append(np.mean(QBER_BB84)); throughputs_bb84.append(THROUGHPUTS_BB84); latency_bb84.append(LATENCY_BB84); loss_bb84.append(LOSS_BB84); rs_bb84.append(RS_BB84)
        skr_b92.append(SECRET_KEY_RATE_B92); qber_b92.append(np.mean(QBER_B92)); throughputs_b92.append(THROUGHPUTS_B92); latency_b92.append(LATENCY_B92); loss_b92.append(LOSS_B92); rs_b92.append(RS_B92)
        skr_cow.append(SECRET_KEY_RATE_COW); qber_cow.append(np.mean(QBER_COW)); throughputs_cow.append(THROUGHPUTS_COW); latency_cow.append(LATENCY_COW); loss_cow.append(LOSS_COW); rs_cow.append(RS_COW); visibility_cow.append(np.mean(VISIBILITY_COW))
        
        skr_bb84e.append(SECRET_KEY_RATE_BB84e); qber_bb84e.append(np.mean(QBER_BB84e)); throughputs_bb84e.append(THROUGHPUTS_BB84e); latency_bb84e.append(LATENCY_BB84e); loss_bb84e.append(LOSS_BB84e); rs_bb84e.append(RS_BB84e)
        skr_b92e.append(SECRET_KEY_RATE_B92e); qber_b92e.append(np.mean(QBER_B92e)); throughputs_b92e.append(THROUGHPUTS_B92e); latency_b92e.append(LATENCY_B92e); loss_b92e.append(LOSS_B92e); rs_b92e.append(RS_B92e)
        skr_cowe.append(SECRET_KEY_RATE_COWe); qber_cowe.append(np.mean(QBER_COWe)); throughputs_cowe.append(THROUGHPUTS_COWe); latency_cowe.append(LATENCY_COWe); loss_cowe.append(LOSS_COWe); rs_cowe.append(RS_COWe); visibility_cowe.append(np.mean(VISIBILITY_COWe))
        
        print()
        print("Simulation "+str((k/keysize_list[-1])*100)+'% completed')
        
    metrics = {
        "keysize":        np.array(keysize_list),
        "R_sk-BB84":       np.array(skr_bb84),
        "QBER-BB84":       qber_bb84,
        "Throughputs-BB84": np.array(throughputs_bb84),
        "Latency-BB84": np.array(latency_bb84),
        "Loss-BB84": np.array(loss_bb84),
        "R_s-BB84": np.array(rs_bb84),
        "R_sk-B92":        np.array(skr_b92),
        "QBER-B92":        qber_b92,
        "Throughputs-B92": np.array(throughputs_b92),
        "Latency-B92": np.array(latency_b92),
        "Loss-B92": np.array(loss_b92),
        "R_s-B92": np.array(rs_b92),
        "R_sk-COW":        np.array(skr_cow),
        "QBER-COW":        qber_cow,
        "Throughputs-COW": np.array(throughputs_cow),
        "Latency-COW": np.array(latency_cow),
        "Loss-COW": np.array(loss_cow),
        "R_s-COW": np.array(rs_cow),
        "Visibility-COW": np.array(visibility_cow),
        "R_sk-BB84+Eve":   np.array(skr_bb84e),
        "QBER-BB84+Eve":   qber_bb84e,
        "Throughputs-BB84+Eve": np.array(throughputs_bb84e),
        "Latency-BB84+Eve": np.array(latency_bb84e),
        "Loss-BB84+Eve": np.array(loss_bb84e),
        "R_s-BB84+Eve": np.array(rs_bb84e),
        "R_sk-B92+Eve":    np.array(skr_b92e),
        "QBER-B92+Eve":    qber_b92e,
        "Throughputs-B92+Eve": np.array(throughputs_b92e),
        "Latency-B92+Eve": np.array(latency_b92e),
        "Loss-B92+Eve": np.array(loss_b92e),
        "R_s-B92+Eve": np.array(rs_b92e),
        "R_sk-COW+Eve":    np.array(skr_cowe),
        "QBER-COW+Eve":    qber_cowe,
        "Throughputs-COW+Eve": np.array(throughputs_cowe),
        "Latency-COW+Eve": np.array(latency_cowe),
        "Loss-COW+Eve": np.array(loss_cowe),
        "R_s-COW+Eve": np.array(rs_cowe),
        "Visibility-COW+Eve": np.array(visibility_cowe)
    }
    pd.DataFrame(metrics).to_csv('metrics_variable-keysize.csv', index=False)

def plot_graph(skr, skr_Eve, qber, qber_Eve, rs, rs_Eve, x_list, x_label, title):
    """ Function that generates the graphs.

    Attributes:
        skr = [skr_list_bb84, skr_list_b92, skr_list_cow]
        skr_Eve = [skr_list_bb84_Eve, skr_list_b92_Eve, skr_list_cow_Eve]
        qber = [qber_list_bb84, qber_list_b92, qber_list_cow]
        qber_Eve = [qber_list_bb84_Eve, qber_list_b92_Eve, qber_list_cow_Eve]
        rs = [rs_list_bb84, rs_list_b92, rs_list_cow]
        rs_Eve = [rs_list_bb84_Eve, rs_list_b92_Eve, rs_list_cow_Eve]
        x_list: List of values ​​for the X-axis. In this simulation, it could be the distance or the key size.
        x_label: X-axis label ("Distance (d) [m]")
        title: f"Aten.={att_lim} dB/m, Keysize={keysize} bits" | f"Aten.={att_lim} dB/m, Distance={distance} meters"
    """
    # display our collected metrics
    # Ideal scenario
    # R_sk(x) and QBER(x)
    fig, ax1 = plt.subplots(figsize=(14, 5))
    linha_y1, = ax1.plot(np.array(x_list), safe_log10(skr[0]), linestyle='-', color='blue', label="R_sk of the BB84")
    linha_y2, = ax1.plot(np.array(x_list), safe_log10(skr[1]), linestyle='-', color='red', label="R_sk of the B92")
    linha_y3, = ax1.plot(np.array(x_list), safe_log10(skr[2]), linestyle='-', color='green', label="R_sk of the COW")
    ax1.set_xlabel(x_label)
    ax1.set_ylabel("log₁₀ Secret Key Rate (R_sk) [bits per sent qubit]")
    ax1.set_title(title)

    ax2 = ax1.twinx()
    linha_z1, = ax2.plot(np.array(x_list), qber[0], linestyle='--', color='orange', label="QBER of the BB84")
    linha_z2, = ax2.plot(np.array(x_list), qber[1], linestyle='--', color='yellow', label="QBER of the B92")
    linha_z3, = ax2.plot(np.array(x_list), qber[2], linestyle='--', color='black', label="QBER of the COW")
    ax2.set_ylabel("QBER")

    linhas = [linha_y1, linha_y2, linha_y3, linha_z1, linha_z2, linha_z3]
    labels = [l.get_label() for l in linhas]
    ax1.legend(linhas, labels, loc="best")

    plt.savefig("graph-ideal_scenario.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # R_s(x)
    fig, ax1 = plt.subplots(figsize=(14, 5))
    linha_y1, = ax1.plot(np.array(x_list), np.array(rs[0])*100, linestyle='-', color='blue', label="BB84")
    linha_y2, = ax1.plot(np.array(x_list), np.array(rs[1])*100, linestyle='-', color='red', label="B92")
    linha_y3, = ax1.plot(np.array(x_list), np.array(rs[2])*100, linestyle='-', color='green', label="COW")
    ax1.set_xlabel(x_label)
    ax1.set_ylabel("R_s - Useful bit rate [%]")
    ax1.set_title(title)

    linhas = [linha_y1, linha_y2, linha_y3]
    labels = [l.get_label() for l in linhas]
    ax1.legend(linhas, labels, loc="best")

    plt.savefig("graph-ideal_scenario-R_s.png", dpi=300, bbox_inches='tight')
    plt.close()


    # Scenario with Eve
    # R_sk(x) and QBER(x)
    fig, ax1 = plt.subplots(figsize=(14, 5))
    linha_y1, = ax1.plot(np.array(x_list), safe_log10(skr_Eve[0]), linestyle='-', color='blue', label="R_sk of the BB84+Eve")
    linha_y2, = ax1.plot(np.array(x_list), safe_log10(skr_Eve[1]), linestyle='-', color='red', label="R_sk of the B92+Eve")
    linha_y3, = ax1.plot(np.array(x_list), safe_log10(skr_Eve[2]), linestyle='-', color='green', label="R_sk of the COW+Eve")
    ax1.set_xlabel(x_label)
    ax1.set_ylabel("log₁₀ Secret Key Rate (R_sk) [bits per sent qubit]")
    ax1.set_title(title)

    ax2 = ax1.twinx()
    linha_z1, = ax2.plot(np.array(x_list), qber_Eve[0], linestyle='--', color='orange', label="QBER of the BB84+Eve")
    linha_z2, = ax2.plot(np.array(x_list), qber_Eve[1], linestyle='--', color='yellow', label="QBER of the B92+Eve")
    linha_z3, = ax2.plot(np.array(x_list), qber_Eve[2], linestyle='--', color='black', label="QBER of the COW+Eve")
    ax2.set_ylabel("QBER")

    linhas = [linha_y1, linha_y2, linha_y3, linha_z1, linha_z2, linha_z3]
    labels = [l.get_label() for l in linhas]
    ax1.legend(linhas, labels, loc="best")

    plt.savefig("graph-Eve_scenario.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # R_s(x)
    fig, ax1 = plt.subplots(figsize=(14, 5))
    linha_y1, = ax1.plot(np.array(x_list), np.array(rs_Eve[0])*100, linestyle='-', color='blue', label="BB84+Eve")
    linha_y2, = ax1.plot(np.array(x_list), np.array(rs_Eve[1])*100, linestyle='-', color='red', label="B92+Eve")
    linha_y3, = ax1.plot(np.array(x_list), np.array(rs_Eve[2])*100, linestyle='-', color='green', label="COW+Eve")
    ax1.set_xlabel(x_label)
    ax1.set_ylabel("R_s - Useful bit rate [%]")
    ax1.set_title(title)

    linhas = [linha_y1, linha_y2, linha_y3]
    labels = [l.get_label() for l in linhas]
    ax1.legend(linhas, labels, loc="best")

    plt.savefig("graph-Eve_scenario-R_s.png", dpi=300, bbox_inches='tight')
    plt.close()

def run_simulation():
    ls_params = {"frequency": 8e6, "wavelength":780, "mean_photon_num": 0.5}
    detector_params = [{"efficiency": 0.65, "dark_count": 100, "time_resolution": 1000, "count_rate": 20e6},
                       {"efficiency": 0.65, "dark_count": 100, "time_resolution": 1000, "count_rate": 20e6}]
    detector_params_cow = [{"efficiency": 0.65, "dark_count": 100, "time_resolution": 1000, "count_rate": 20e6},
                       {"efficiency": 0.65, "dark_count": 100, "time_resolution": 1000, "count_rate": 20e6},
                       {"efficiency": 0.65, "dark_count": 100, "time_resolution": 1000, "count_rate": 20e6}]
    keysize = 10000
    # channel_parameters = (distance [in meters], attenuation [in dB/m], polarization_fidelity [in %])
    channel_parameters = (700, 0.0002, 0.97)
    sim_variable_distance(d_step=1000, d_lim=100000, channel_parameters=channel_parameters, ls_params=ls_params, detector_params=detector_params, detector_params_cow=detector_params_cow, keysize=keysize)
    sim_variable_keysize(keysize_list=[20, 50, 100, 200, 400, 800, 1600, 5000, 20000, 40000], channel_parameters, ls_params, detector_params, detector_params_cow):
    
    
    df_d = pd.read_csv('metrics_variable-distance.csv')
    df_k = pd.read_csv('metrics_variable-keysize.csv')
    
    
    
    plot_graph(skr=[df_d["R_sk-BB84"], df_d["R_sk-B92"], df_d["R_sk-COW"]], 
           skr_Eve=[df_d["R_sk-BB84+Eve"], df_d["R_sk-B92+Eve"], df_d["R_sk-COW+Eve"]], 
           qber=[df_d["QBER-BB84"], df_d["QBER-B92"], df_d["QBER-COW"]], 
           qber_Eve=[df_d["QBER-BB84+Eve"], df_d["QBER-B92+Eve"], df_d["QBER-COW+Eve"]], 
           rs=[df_d["R_s-BB84"], df_d["R_s-B92"], df_d["R_s-COW"]], 
           rs_Eve=[df_d["R_s-BB84+Eve"], df_d["R_s-B92+Eve"], df_d["R_s-COW+Eve"]], 
           x_list=df_d["distance"], 
           x_label="Distance (d) [m]", title=f"Aten.={channel_parameters[1]} dB/m, Keysize={keysize} bits")
    plot_graph(skr=[df_k["R_sk-BB84"], df_k["R_sk-B92"], df_k["R_sk-COW"]], 
           skr_Eve=[df_k["R_sk-BB84+Eve"], df_k["R_sk-B92+Eve"], df_k["R_sk-COW+Eve"]], 
           qber=[df_k["QBER-BB84"], df_k["QBER-B92"], df_k["QBER-COW"]], 
           qber_Eve=[df_k["QBER-BB84+Eve"], df_k["QBER-B92+Eve"], df_k["QBER-COW+Eve"]], 
           rs=[df_k["R_s-BB84"], df_k["R_s-B92"], df_k["R_s-COW"]], 
           rs_Eve=[df_k["R_s-BB84+Eve"], df_k["R_s-B92+Eve"], df_k["R_s-COW+Eve"]], 
           x_list=df_k["keysize"], 
           x_label="Key Size (k) [bit width]", title=f"Aten.={channel_parameters[1]} dB/m, Distance={channel_parameters[0]} meters")

if __name__ == "__main__":
    run_simulation()
