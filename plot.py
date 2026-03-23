import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

def safe_log10(lst: list) -> np.ndarray:
    arr = np.array(lst, dtype=float)
    arr[arr <= 0] = np.nan
    return np.log10(arr)

def plot_graph(skr, skr_Eve, qber, qber_Eve, rs, rs_Eve, x_list, x_label, title, filename):
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
    # R_sk(x)
    fig, (ax1, ax3) = plt.subplots(2, figsize=(12, 6), sharex=True)
    fig.suptitle(title)
    linha_y1, = ax1.plot(np.array(x_list), safe_log10(skr[0]), linestyle=(0, (1, 1)), color='blue', label="R_sk of the BB84")
    linha_y2, = ax1.plot(np.array(x_list), safe_log10(skr[1]), linestyle=(0, (1, 5)), color='green', label="R_sk of the B92")
    linha_y3, = ax1.plot(np.array(x_list), safe_log10(skr[2]), linestyle=(0, (1, 10)), color='red', label="R_sk of the COW")
    ax1.set_ylabel("log₁₀ Secret Key Rate (R_sk)\n[bits per sent qubit]")
    # QBER(x)
    ax2 = ax1.twinx()
    linha_z1, = ax2.plot(np.array(x_list), np.array(qber[0])*100, linestyle=(0, (5, 1)), color='orange', label="QBER of the BB84")
    linha_z2, = ax2.plot(np.array(x_list), np.array(qber[1])*100, linestyle=(0, (5, 5)), color='maroon', label="QBER of the B92")
    linha_z3, = ax2.plot(np.array(x_list), np.array(qber[2])*100, linestyle=(0, (5, 10)), color='black', label="QBER of the COW")
    ax2.set_ylabel("QBER [%]")
    # R_s(x)
    linha_w1, = ax3.plot(np.array(x_list), np.array(rs[0])*100, linestyle="solid", color='grey', label="BB84")
    linha_w2, = ax3.plot(np.array(x_list), np.array(rs[1])*100, linestyle=(0, (3, 1, 1, 1)), color='cyan', label="B92")
    linha_w3, = ax3.plot(np.array(x_list), np.array(rs[2])*100, linestyle=(0, (3, 1, 1, 1, 1, 1)), color='violet', label="COW")
    ax3.set_xlabel(x_label)
    ax3.set_ylabel("R_s - Useful bit rate [%]")

    linhas = [linha_y1, linha_y2, linha_y3, linha_z1, linha_z2, linha_z3, linha_w1, linha_w2, linha_w3]
    labels = [l.get_label() for l in linhas]
    ax3.legend(linhas, labels, loc='upper center', bbox_to_anchor=(0.5, -0.2),
          fancybox=True, shadow=True, ncol=5)
    ax1.grid(True)
    ax3.grid(True)
    plt.savefig(f"{filename}_graph-ideal_scenario.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Scenario with Eve
    # R_sk(x)
    fig, (ax1, ax3) = plt.subplots(2, figsize=(12, 6), sharex=True)
    fig.suptitle(title)
    linha_y1, = ax1.plot(np.array(x_list), safe_log10(skr_Eve[0]), linestyle=(0, (1, 1)), color='blue', label="R_sk of the BB84+Eve")
    linha_y2, = ax1.plot(np.array(x_list), safe_log10(skr_Eve[1]), linestyle=(0, (1, 5)), color='green', label="R_sk of the B92+Eve")
    linha_y3, = ax1.plot(np.array(x_list), safe_log10(skr_Eve[2]), linestyle=(0, (1, 10)), color='red', label="R_sk of the COW+Eve")
    ax1.set_ylabel("log₁₀ Secret Key Rate (R_sk)\n[bits per sent qubit]")
    # QBER(x)
    ax2 = ax1.twinx()
    linha_z1, = ax2.plot(np.array(x_list), np.array(qber_Eve[0])*100, linestyle=(0, (5, 1)), color='orange', label="QBER of the BB84+Eve")
    linha_z2, = ax2.plot(np.array(x_list), np.array(qber_Eve[1])*100, linestyle=(0, (5, 5)), color='maroon', label="QBER of the B92+Eve")
    linha_z3, = ax2.plot(np.array(x_list), np.array(qber_Eve[2])*100, linestyle=(0, (5, 10)), color='black', label="QBER of the COW+Eve")
    ax2.set_ylabel("QBER [%]")
    # R_s(x)
    linha_w1, = ax3.plot(np.array(x_list), np.array(rs_Eve[0])*100, linestyle="solid", color='grey', label="BB84+Eve")
    linha_w2, = ax3.plot(np.array(x_list), np.array(rs_Eve[1])*100, linestyle=(0, (3, 1, 1, 1)), color='cyan', label="B92+Eve")
    linha_w3, = ax3.plot(np.array(x_list), np.array(rs_Eve[2])*100, linestyle=(0, (3, 1, 1, 1, 1, 1)), color='violet', label="COW+Eve")
    ax3.set_xlabel(x_label)
    ax3.set_ylabel("R_s - Useful bit rate [%]")

    linhas = [linha_y1, linha_y2, linha_y3, linha_z1, linha_z2, linha_z3, linha_w1, linha_w2, linha_w3]
    labels = [l.get_label() for l in linhas]
    ax3.legend(linhas, labels, loc='upper center', bbox_to_anchor=(0.5, -0.2),
          fancybox=True, shadow=True, ncol=5)
    ax1.grid(True)
    ax3.grid(True)
    plt.savefig(f"{filename}_graph-Eve_scenario.png", dpi=300, bbox_inches='tight')
    plt.close()
    
def plot_v(visibility, visibility_Eve, x_list, x_label, title, filename):
    """ Function that generates the graphs.

    Attributes:
        visibility = visibility_list_cow
        visibility_Eve = visibility_list_cow_Eve
        x_list: List of values ​​for the X-axis. In this simulation, it could be the distance or the key size.
        x_label: X-axis label ("Distance (d) [m]")
        title: f"Aten.={att_lim} dB/m, Keysize={keysize} bits" | f"Aten.={att_lim} dB/m, Distance={distance} meters"
    """
    # display our collected metrics
    # Ideal scenario
    # R_sk(x)
    fig, ax1 = plt.subplots(figsize=(12, 6))
    fig.suptitle(title)
    linha_y1, = ax1.plot(np.array(x_list), visibility, linestyle=(0, (1, 1)), color='blue', label="V of the COW")
    linha_y2, = ax1.plot(np.array(x_list), visibility_Eve, linestyle=(0, (3, 1, 1, 1, 1, 1)), color='green', label="V of the COW+Eve")
    ax1.set_xlabel(x_label)
    ax1.set_ylabel("V - Visibility [%]")

    linhas = [linha_y1, linha_y2]
    labels = [l.get_label() for l in linhas]
    ax1.legend(linhas, labels, loc='upper center', bbox_to_anchor=(0.5, -0.2),
          fancybox=True, shadow=True, ncol=5)
    ax1.grid(True)
    plt.savefig(f"{filename}_graph-visibility.png", dpi=300, bbox_inches='tight')
    plt.close()
    
def main():
    # channel_parameters = (distance [in meters], attenuation [in dB/m], polarization_fidelity [in %])
    channel_parameters = (700, 0.0002, 0.97)
    keysize = 10000
    df_d = pd.read_csv('data/metrics_variable-distance.csv')
    df_k = pd.read_csv('data/metrics_variable-keysize.csv')

    plot_graph(skr=[df_d["R_sk-BB84"], df_d["R_sk-B92"], df_d["R_sk-COW"]], 
           skr_Eve=[df_d["R_sk-BB84+Eve"], df_d["R_sk-B92+Eve"], df_d["R_sk-COW+Eve"]], 
           qber=[df_d["QBER-BB84"], df_d["QBER-B92"], df_d["QBER-COW"]], 
           qber_Eve=[df_d["QBER-BB84+Eve"], df_d["QBER-B92+Eve"], df_d["QBER-COW+Eve"]], 
           rs=[df_d["R_s-BB84"], df_d["R_s-B92"], df_d["R_s-COW"]], 
           rs_Eve=[df_d["R_s-BB84+Eve"], df_d["R_s-B92+Eve"], df_d["R_s-COW+Eve"]], 
           x_list=df_d["distance"], 
           x_label="Distance (d) [m]", title=f"Aten.={channel_parameters[1]} dB/m, Keysize={keysize} bits", filename="distance")

    plot_graph(skr=[df_k["R_sk-BB84"], df_k["R_sk-B92"], df_k["R_sk-COW"]], 
           skr_Eve=[df_k["R_sk-BB84+Eve"], df_k["R_sk-B92+Eve"], df_k["R_sk-COW+Eve"]], 
           qber=[df_k["QBER-BB84"], df_k["QBER-B92"], df_k["QBER-COW"]], 
           qber_Eve=[df_k["QBER-BB84+Eve"], df_k["QBER-B92+Eve"], df_k["QBER-COW+Eve"]], 
           rs=[df_k["R_s-BB84"], df_k["R_s-B92"], df_k["R_s-COW"]], 
           rs_Eve=[df_k["R_s-BB84+Eve"], df_k["R_s-B92+Eve"], df_k["R_s-COW+Eve"]], 
           x_list=df_k["keysize"], 
           x_label="Key Size (k) [bit width]", title=f"Aten.={channel_parameters[1]} dB/m, Distance={channel_parameters[0]} meters", filename="keysize")
           
    plot_v(visibility=df_d["Visibility-COW"], 
           visibility_Eve=df_d["Visibility-COW+Eve"],
           x_list=df_d["distance"], 
           x_label="Distance (d) [m]", title=f"Aten.={channel_parameters[1]} dB/m, Keysize={keysize} bits", filename="distance")
    
    plot_v(visibility=df_k["Visibility-COW"], 
           visibility_Eve=df_k["Visibility-COW+Eve"],
           x_list=df_k["keysize"], 
           x_label="Key Size (k) [bit width]", title=f"Aten.={channel_parameters[1]} dB/m, Distance={channel_parameters[0]} meters", filename="keysize")

if __name__ == "__main__":
    main()
