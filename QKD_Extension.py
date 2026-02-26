import math

from ipywidgets import interact
from matplotlib import pyplot as plt

from sequence.components.optical_channel import QuantumChannel, ClassicalChannel
from sequence.kernel.event import Event
from sequence.kernel.process import Process
from sequence.kernel.timeline import Timeline
from sequence.qkd.BB84 import pair_bb84_protocols
from sequence.topology.node import QKDNode
import sequence.utils.log as log
import numpy as np


def binary_entropy(Q):
    if Q == 0 or Q == 1:
        return 0
    return -Q * math.log2(Q) - (1 - Q) * math.log2(1 - Q)


def simulation_BB84(runtime=20, log_filename=-1, distance=1e3, polarization_fidelity=0.97, attenuation=0.0002, keysize=256, key_num=math.inf):
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
    ls_params = {"frequency": 80e6, "mean_photon_num": 0.1}
    alice = QKDNode("alice", tl, stack_size=1)
    alice.set_seed(0)

    for name, param in ls_params.items():
        alice.update_lightsource_params(name, param)

    # Bob
    detector_params = [{"efficiency": 0.8, "dark_count": 10, "time_resolution": 10, "count_rate": 50e6},
                       {"efficiency": 0.8, "dark_count": 10, "time_resolution": 10, "count_rate": 50e6}]
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

    secret_key_rate_mean = 0

    QBER = alice.protocol_stack[0].error_rates
    THROUGHPUTS = alice.protocol_stack[0].throughputs
    LATENCY = alice.protocol_stack[0].latency
    for i, e in enumerate(alice.protocol_stack[0].error_rates):
        R_s = alice.protocol_stack[0].sifted_bits_length[i]/len(alice.protocol_stack[0].bit_lists[0])
        h_Q = binary_entropy(e)
        R_sk = R_s*(1-h_Q)
        secret_key_rate_mean += R_sk
    SECRET_KEY_RATE = secret_key_rate_mean/len(alice.protocol_stack[0].error_rates)
    LOSS = 1-10**((distance*attenuation)/(-10))
    
    return QBER, THROUGHPUTS, LATENCY, SECRET_KEY_RATE, LOSS


def plot_graph():
    # Gráfico Taxa de chave secreta em função da distância
    d_list = []
    skr_list = []
    qber_list = []
    d_step = 100
    d_lim = 10000 # Em metros
    att_lim = 0.0002 # dB/metro
    keysize = 25
    d = 0
    while d <= d_lim:
        QBER, THROUGHPUTS, LATENCY, SECRET_KEY_RATE, LOSS = simulation_BB84(distance=d, attenuation=att_lim, keysize=keysize)
        d_list.append(d)
        skr_list.append(SECRET_KEY_RATE)
        qber_list.append(np.mean(QBER))
        print()
        print(str((d/d_lim)*100)+'% concluído')
        d += d_step
        
    skr_array = np.log10(np.array(skr_list))
    d_array = np.array(d_list)
    qber_array= np.array(qber_list)
    
    # display our collected metrics
    fig, ax1 = plt.subplots()
    linha_y, = ax1.plot(d_array, skr_array, linestyle='-', color='blue', label="R_sk(d)")
    ax1.set_xlabel("Distance (d) [m]")
    ax1.set_ylabel("Log_10 Secret Key Rate (R_sk) [bits per sent qubit]")
    ax1.set_title("Attenuation [dB/m]:"+str(att_lim)+", Keysize [bits of width]:"+str(keysize))

    ax2 = ax1.twinx()
    linha_z, = ax2.plot(d_array, qber_array, linestyle='--', color='red', label="QBER(d)")
    ax2.set_ylabel("QBER")

    linhas = [linha_y, linha_z]
    labels = [l.get_label() for l in linhas]
    ax1.legend(linhas, labels, loc="best")

    plt.savefig("graph.png", dpi=300, bbox_inches='tight')
    plt.show()

def run_simulation():
    # constants
    #log_filename = "bb84.log"
    #interactive_plot = interact(simulation_BB84, runtime=(1, 10, 20), attenuation=(0.0002, 0.002, 0.02), keysize=[128, 256, 512])
    #interactive_plot
    plot_graph()

if __name__ == "__main__":
    run_simulation()

