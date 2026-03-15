import math

from matplotlib import pyplot as plt

from sequence.components.optical_channel import QuantumChannel, ClassicalChannel
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


def binary_entropy(Q):
    if Q == 0 or Q == 1:
        return 0
    return -Q * math.log2(Q) - (1 - Q) * math.log2(1 - Q)
    
def _collect_metrics(protocol, distance: float, attenuation: float):
    """Extrai QBER, throughput, latência, SKR e perda de um protocolo."""
    secret_key_rate_mean = 0

    QBER = protocol.error_rates
    THROUGHPUTS = protocol.throughputs
    LATENCY = protocol.latency
    for i, e in enumerate(QBER):
        R_s = protocol.sifted_bits_length[i] / len(protocol.bit_lists[0])
        secret_key_rate_mean += R_s * (1 - binary_entropy(e))
    SECRET_KEY_RATE = secret_key_rate_mean/len(QBER)
    LOSS = 1-10**((distance*attenuation)/(-10))
    return QBER, THROUGHPUTS, LATENCY, SECRET_KEY_RATE, LOSS


def simulation_BB84(ls_params, detector_params, runtime=20, log_filename=-1, distance=1e3, polarization_fidelity=0.97, attenuation=0.0002, keysize=256, key_num=math.inf):
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
    alice = QKDNode("alice", tl, stack_size=1)
    alice.set_seed(0)

    for name, param in ls_params.items():
        alice.update_lightsource_params(name, param)

    # Bob
    bob = QKDNode("bob", tl, stack_size=1)
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

def simulation_B92(ls_params, detector_params, runtime=20, log_filename=-1, distance=1e3, polarization_fidelity=0.97, attenuation=0.0002, keysize=256, key_num=math.inf):
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
    alice = QKDNode("alice", tl, stack_size=1, qkdtype=1)
    alice.set_seed(0)

    for name, param in ls_params.items():
        alice.update_lightsource_params(name, param)

    # Bob
    bob = QKDNode("bob", tl, stack_size=1, qkdtype=1)
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
    alice = QKDNode("alice", tl, encoding=time_bin_cow, stack_size=1, qkdtype=2)
    alice.set_seed(0)

    for name, param in ls_params.items():
        alice.update_lightsource_params(name, param)

    # Bob
    bob = QKDNode("bob", tl, encoding=time_bin_cow, stack_size=1, qkdtype=2)
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

    QBER, THROUGHPUTS, LATENCY, SKR, LOSS = _collect_metrics(alice.protocol_stack[0], distance, attenuation)
    VISIBILITY = bob.protocol_stack[0].visibility
    return QBER, THROUGHPUTS, LATENCY, SKR, LOSS, VISIBILITY
    
    
def simulation_BB84_Eve(ls_params, detector_params, runtime=20, log_filename=-1, distance=1e3, polarization_fidelity=0.97, attenuation=0.0002, keysize=256, key_num=math.inf, eve_intercept_rate = 1.0, eve_position = 0.5):
    dist_ae = distance * eve_position
    dist_eb = distance * (1.0 - eve_position)

    tl = Timeline(runtime * 1e9)
    tl.show_progress = True
    if log_filename != -1:
        log.set_logger(__name__, tl, log_filename)
        log.set_logger_level('DEBUG')
        log.track_module('BB84')
        log.track_module('light_source')
    
    qc_ae = QuantumChannel("qc_ae", tl, distance=dist_ae, polarization_fidelity=polarization_fidelity, attenuation=attenuation)
    qc_eb = QuantumChannel("qc_eb", tl, distance=dist_eb, polarization_fidelity=polarization_fidelity, attenuation=attenuation)
    qc_ba = QuantumChannel("qc_ba", tl, distance=distance, polarization_fidelity=polarization_fidelity, attenuation=attenuation)
    
    cc_ab = ClassicalChannel("cc_ab", tl, distance=distance)
    cc_ba = ClassicalChannel("cc_ba", tl, distance=distance)
    cc_ab.delay += 1e9   # 1 ms
    cc_ba.delay += 1e9
    
    alice = QKDNode("alice", tl, stack_size=1)
    alice.set_seed(0)
    for name, param in ls_params.items():
        alice.update_lightsource_params(name, param)

    eve = EveNode("eve", tl, encoding=polarization, intercept_rate=eve_intercept_rate, seed=2)

    bob = QKDNode("bob", tl, stack_size=1)
    bob.set_seed(1)
    for i, dp in enumerate(detector_params):
        for name, param in dp.items():
            bob.update_detector_params(i, name, param)

    qc_ae.set_ends(alice, eve.name)
    eve.destination = bob.name
    qc_eb.set_ends(eve, bob.name)
    qc_ba.set_ends(bob, alice.name)
    cc_ab.set_ends(alice, bob.name)
    cc_ba.set_ends(bob, alice.name)
    
    pair_bb84_protocols(alice.protocol_stack[0], bob.protocol_stack[0])
    tl.schedule(Event(0, Process(alice.protocol_stack[0], "push", [keysize, key_num, 6e12])))
    tl.init()
    tl.run()
    return _collect_metrics(alice.protocol_stack[0], distance, attenuation)
    

def simulation_B92_Eve(ls_params, detector_params, runtime=20, log_filename=-1, distance=1e3, polarization_fidelity=0.97, attenuation=0.0002, keysize=256, key_num=math.inf, eve_intercept_rate = 1.0, eve_position = 0.5):
    dist_ae = distance * eve_position
    dist_eb = distance * (1.0 - eve_position)

    tl = Timeline(runtime * 1e9)
    tl.show_progress = True
    if log_filename != -1:
        log.set_logger(__name__, tl, log_filename)
        log.set_logger_level('DEBUG')
        log.track_module('B92')
        log.track_module('light_source')

    qc_ae = QuantumChannel("qc_ae", tl, distance=dist_ae, polarization_fidelity=polarization_fidelity, attenuation=attenuation)
    qc_eb = QuantumChannel("qc_eb", tl, distance=dist_eb, polarization_fidelity=polarization_fidelity, attenuation=attenuation)
    qc_ba = QuantumChannel("qc_ba", tl, distance=distance, polarization_fidelity=polarization_fidelity, attenuation=attenuation)

    cc_ab = ClassicalChannel("cc_ab", tl, distance=distance)
    cc_ba = ClassicalChannel("cc_ba", tl, distance=distance)
    cc_ab.delay += 1e9
    cc_ba.delay += 1e9

    alice = QKDNode("alice", tl, stack_size=1, qkdtype=1)
    alice.set_seed(0)
    for name, param in ls_params.items():
        alice.update_lightsource_params(name, param)

    eve = EveNode("eve", tl, encoding=polarization, intercept_rate=eve_intercept_rate, seed=2)

    bob = QKDNode("bob", tl, stack_size=1, qkdtype=1)
    bob.set_seed(1)
    for i, dp in enumerate(detector_params):
        for name, param in dp.items():
            bob.update_detector_params(i, name, param)

    qc_ae.set_ends(alice, eve.name)
    eve.destination = bob.name
    qc_eb.set_ends(eve, bob.name)
    qc_ba.set_ends(bob, alice.name)
    cc_ab.set_ends(alice, bob.name)
    cc_ba.set_ends(bob, alice.name)

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

    qc_ae = QuantumChannel("qc_ae", tl, distance=dist_ae, polarization_fidelity=polarization_fidelity, attenuation=attenuation)
    qc_eb = QuantumChannel("qc_eb", tl, distance=dist_eb, polarization_fidelity=polarization_fidelity, attenuation=attenuation)
    qc_ba = QuantumChannel("qc_ba", tl, distance=distance, polarization_fidelity=polarization_fidelity, attenuation=attenuation)
    cc_ab = ClassicalChannel("cc_ab", tl, distance=distance)
    cc_ba = ClassicalChannel("cc_ba", tl, distance=distance)
    cc_ab.delay += 1e9
    cc_ba.delay += 1e9

    alice = QKDNode("alice", tl, encoding=time_bin_cow, stack_size=1, qkdtype=2)
    alice.set_seed(0)
    for name, param in ls_params.items():
        alice.update_lightsource_params(name, param)

    eve = EveNode("eve", tl, encoding=time_bin_cow, intercept_rate=eve_intercept_rate, seed=2)
    bob = QKDNode("bob", tl, encoding=time_bin_cow, stack_size=1, qkdtype=2)
    bob.set_seed(1)
    for i, dp in enumerate(detector_params):
        for name, param in dp.items():
            bob.update_detector_params(i, name, param)

    qc_ae.set_ends(alice, eve.name)
    eve.destination = bob.name
    qc_eb.set_ends(eve, bob.name)
    qc_ba.set_ends(bob, alice.name)
    cc_ab.set_ends(alice, bob.name)
    cc_ba.set_ends(bob, alice.name)

    pair_cow_protocols(alice.protocol_stack[0], bob.protocol_stack[0])
    tl.schedule(Event(0, Process(alice.protocol_stack[0], "push", [keysize, key_num, 6e12])))
    tl.init()
    tl.run()
    
    QBER, THROUGHPUTS, LATENCY, SKR, LOSS = _collect_metrics(alice.protocol_stack[0], distance, attenuation)
    VISIBILITY = bob.protocol_stack[0].visibility
    return QBER, THROUGHPUTS, LATENCY, SKR, LOSS, VISIBILITY

# plot_graph (
#            d_step = step of the distance (in meters),
#            d_lim = limit distance (in meters)
#            att_lim = attenuation limit (in dB/meters)
#            keysize = key size (in number of logical bits)):
def plot_graph(d_step, d_lim, att_lim, keysize):
    d_list = []

    ls_params = {"frequency": 3.8e+14, "wavelength":795, "mean_photon_num": 0.1}
    detector_params = [{"efficiency": 0.65, "dark_count": 100, "time_resolution": 1000, "count_rate": 20e6},
                       {"efficiency": 0.65, "dark_count": 100, "time_resolution": 1000, "count_rate": 20e6}]
    detector_params_cow = [{"efficiency": 0.65, "dark_count": 100, "time_resolution": 1000, "count_rate": 20e6},
                       {"efficiency": 0.65, "dark_count": 100, "time_resolution": 1000, "count_rate": 20e6},
                       {"efficiency": 0.65, "dark_count": 100, "time_resolution": 1000, "count_rate": 20e6}]

    skr_bb84, qber_bb84, throughputs_bb84, latency_bb84, loss_bb84 = [], [], [], [], []
    skr_b92, qber_b92, throughputs_b92, latency_b92, loss_b92 = [], [], [], [], []
    skr_cow, qber_cow, throughputs_cow, latency_cow, loss_cow, visibility_cow = [], [], [], [], [], []
    
    skr_bb84e, qber_bb84e, throughputs_bb84e, latency_bb84e, loss_bb84e = [], [], [], [], []
    skr_b92e, qber_b92e, throughputs_b92e, latency_b92e, loss_b92e = [], [], [], [], []
    skr_cowe, qber_cowe, throughputs_cowe, latency_cowe, loss_cowe, visibility_cowe = [], [], [], [], [], []


    d = 0
    while d <= d_lim:
        # Sem Eve (Cenário Ideal)
        QBER_BB84, THROUGHPUTS_BB84, LATENCY_BB84, SECRET_KEY_RATE_BB84, LOSS_BB84 = simulation_BB84(ls_params, detector_params, distance=d, attenuation=att_lim, keysize=keysize)
        QBER_B92, THROUGHPUTS_B92, LATENCY_B92, SECRET_KEY_RATE_B92, LOSS_B92 = simulation_B92(ls_params, detector_params, distance=d, attenuation=att_lim, keysize=keysize)
        QBER_COW, THROUGHPUTS_COW, LATENCY_COW, SECRET_KEY_RATE_COW, LOSS_COW, VISIBILITY_COW = simulation_COW(ls_params, detector_params_cow, distance=d, attenuation=att_lim, keysize=keysize)
        
        # Com Eve
        QBER_BB84e, THROUGHPUTS_BB84e, LATENCY_BB84e, SECRET_KEY_RATE_BB84e, LOSS_BB84e = simulation_BB84_Eve(ls_params, detector_params, distance=d, attenuation=att_lim, keysize=keysize)
        QBER_B92e, THROUGHPUTS_B92e, LATENCY_B92e, SECRET_KEY_RATE_B92e, LOSS_B92e = simulation_B92_Eve(ls_params, detector_params, distance=d, attenuation=att_lim, keysize=keysize)
        QBER_COWe, THROUGHPUTS_COWe, LATENCY_COWe, SECRET_KEY_RATE_COWe, LOSS_COWe, VISIBILITY_COWe = simulation_COW_Eve(ls_params, detector_params_cow, distance=d, attenuation=att_lim, keysize=keysize)
        
        d_list.append(d)
        
        skr_bb84.append(SECRET_KEY_RATE_BB84); qber_bb84.append(np.mean(QBER_BB84)); throughputs_bb84.append(THROUGHPUTS_BB84); latency_bb84.append(LATENCY_BB84); loss_bb84.append(LOSS_BB84)
        skr_b92.append(SECRET_KEY_RATE_B92); qber_b92.append(np.mean(QBER_B92)); throughputs_b92.append(THROUGHPUTS_B92); latency_b92.append(LATENCY_B92); loss_b92.append(LOSS_B92)
        skr_cow.append(SECRET_KEY_RATE_COW); qber_cow.append(np.mean(QBER_COW)); throughputs_cow.append(THROUGHPUTS_COW); latency_cow.append(LATENCY_COW); loss_cow.append(LOSS_COW), visibility_cow.append(np.mean(VISIBILITY_COW))
        
        skr_bb84e.append(SECRET_KEY_RATE_BB84e); qber_bb84e.append(np.mean(QBER_BB84e)); throughputs_bb84e.append(THROUGHPUTS_BB84e); latency_bb84e.append(LATENCY_BB84e); loss_bb84e.append(LOSS_BB84e)
        skr_b92e.append(SECRET_KEY_RATE_B92e); qber_b92e.append(np.mean(QBER_B92e)); throughputs_b92e.append(THROUGHPUTS_B92e); latency_b92e.append(LATENCY_B92e); loss_b92e.append(LOSS_B92e)
        skr_cowe.append(SECRET_KEY_RATE_COWe); qber_cowe.append(np.mean(QBER_COWe)); throughputs_cowe.append(THROUGHPUTS_COWe); latency_cowe.append(LATENCY_COWe); loss_cowe.append(LOSS_COWe), visibility_cowe.append(np.mean(VISIBILITY_COWe))
        
        print()
        print("Simulation "+str((d/d_lim)*100)+'% completed')
        d += d_step
    
    def safe_log10(lst: list) -> np.ndarray:
        arr = np.array(lst, dtype=float)
        arr[arr <= 0] = np.nan
        return np.log10(arr)

    log = {'R_sk - BB84': skr_array[0], "QBER - BB84": qber_array[0], 'R_sk - B92': skr_array[1], "QBER - B92": qber_array[1], 'R_sk - COW': skr_array[2], "QBER - COW": qber_array[2], 'distance': d_array}
    metrics = {
        "distance":        np.array(d_list),
        "R_sk-BB84":       safe_log10(skr_bb84),
        "QBER-BB84":       qber_bb84,
        "Throughputs-BB84": np.array(throughputs_bb84),
        "Latency-BB84": np.array(latency_bb84),
        "Loss-BB84": np.array(loss_bb84),
        "R_sk-B92":        safe_log10(skr_b92),
        "QBER-B92":        qber_b92,
        "Throughputs-B92": np.array(throughputs_b92),
        "Latency-B92": np.array(latency_b92),
        "Loss-B92": np.array(loss_b92),
        "R_sk-COW":        safe_log10(skr_cow),
        "QBER-COW":        qber_cow,
        "Throughputs-COW": np.array(throughputs_cow),
        "Latency-COW": np.array(latency_cow),
        "Loss-COW": np.array(loss_cow),
        "Visibility-COW": np.array(visibility_cow),
        "R_sk-BB84+Eve":   safe_log10(skr_bb84e),
        "QBER-BB84+Eve":   qber_bb84e,
        "Throughputs-BB84+Eve": np.array(throughputs_bb84e),
        "Latency-BB84+Eve": np.array(latency_bb84e),
        "Loss-BB84+Eve": np.array(loss_bb84e),
        "R_sk-B92+Eve":    safe_log10(skr_b92e),
        "QBER-B92+Eve":    qber_b92e,
        "Throughputs-B92+Eve": np.array(throughputs_b92e),
        "Latency-B92+Eve": np.array(latency_b92e),
        "Loss-B92+Eve": np.array(loss_b92e),
        "R_sk-COW+Eve":    safe_log10(skr_cowe),
        "QBER-COW+Eve":    qber_cowe,
        "Throughputs-COW+Eve": np.array(throughputs_cowe),
        "Latency-COW+Eve": np.array(latency_cowe),
        "Loss-COW+Eve": np.array(loss_cowe),
        "Visibility-COW+Eve": np.array(visibility_cowe)
    }
    pd.DataFrame(metrics).to_csv('metrics.csv', index=False)
    
    # display our collected metrics
    # Ideal scenario
    fig, ax1 = plt.subplots(1, 2, figsize=(14, 5))
    linha_y1, = ax1.plot(d_array, safe_log10(skr_bb84), linestyle='-', color='blue', label="R_sk(d) of the BB84")
    linha_y2, = ax1.plot(d_array, safe_log10(skr_b92), linestyle='-', color='red', label="R_sk(d) of the B92")
    linha_y3, = ax1.plot(d_array, safe_log10(skr_cow), linestyle='-', color='green', label="R_sk(d) of the COW")
    ax1.set_xlabel("Distance (d) [m]")
    ax1.set_ylabel("log₁₀ Secret Key Rate (R_sk) [bits per sent qubit]")
    ax1.set_title(f"Aten.={att_lim} dB/m, Keysize={keysize} bits")

    ax2 = ax1.twinx()
    linha_z1, = ax2.plot(d_array, qber_bb84, linestyle='--', color='orange', label="QBER(d) of the BB84")
    linha_z2, = ax2.plot(d_array, qber_b92, linestyle='--', color='yellow', label="QBER(d) of the B92")
    linha_z3, = ax2.plot(d_array, qber_cow, linestyle='--', color='black', label="QBER(d) of the COW")
    ax2.set_ylabel("QBER")

    linhas = [linha_y1, linha_y2, linha_y3, linha_z1, linha_z2, linha_z3]
    labels = [l.get_label() for l in linhas]
    ax1.legend(linhas, labels, loc="best")

    plt.savefig("graph-ideal_scenario.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Scenario with Eve
    fig, ax1 = plt.subplots(1, 2, figsize=(14, 5))
    linha_y1, = ax1.plot(d_array, safe_log10(skr_bb84e), linestyle='-', color='blue', label="R_sk(d) of the BB84+Eve")
    linha_y2, = ax1.plot(d_array, safe_log10(skr_b92e), linestyle='-', color='red', label="R_sk(d) of the B92+Eve")
    linha_y3, = ax1.plot(d_array, safe_log10(skr_cowe), linestyle='-', color='green', label="R_sk(d) of the COW+Eve")
    ax1.set_xlabel("Distance (d) [m]")
    ax1.set_ylabel("log₁₀ Secret Key Rate (R_sk) [bits per sent qubit]")
    ax1.set_title(f"Aten.={att_lim} dB/m, Keysize={keysize} bits")

    ax2 = ax1.twinx()
    linha_z1, = ax2.plot(d_array, qber_bb84e, linestyle='--', color='orange', label="QBER(d) of the BB84+Eve")
    linha_z2, = ax2.plot(d_array, qber_b92e, linestyle='--', color='yellow', label="QBER(d) of the B92+Eve")
    linha_z3, = ax2.plot(d_array, qber_cowe, linestyle='--', color='black', label="QBER(d) of the COW+Eve")
    ax2.set_ylabel("QBER")

    linhas = [linha_y1, linha_y2, linha_y3, linha_z1, linha_z2, linha_z3]
    labels = [l.get_label() for l in linhas]
    ax1.legend(linhas, labels, loc="best")

    plt.savefig("graph-Eve_scenario.png", dpi=300, bbox_inches='tight')
    plt.close()

def run_simulation():
    plot_graph(d_step=100, d_lim=10000, att_lim=0.0002, keysize=25)

if __name__ == "__main__":
    run_simulation()









