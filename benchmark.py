import pickle
import matplotlib.pyplot as plt
import numpy as np
from qutip import fidelity, Qobj
import os
from scipy.interpolate import RegularGridInterpolator
import time
import sven_solution as ss
import numpy as np
import max_solution as ms
import niko_solution as ns
from bm_util import *

SYNTHETIC_DATA = ['noisy_wigner_' + str(i) + '.pickle' for i in range(16)]

SYNTHETIC_EXPECTED_OUTCOME = ['quantum_state_' + str(i) + '.pickle' for i in range(8)]

METHOD = [("Matrix representation", "mat_repr", ms.reconstruct_density_matrix_robust), ("Least Square optimization", "ls_opti", ns.ls_reconstruct), ("Wigner projection tomography", "wigner_p_topo", ss.sven_reconstruct)]

def benchmark():
    """
    Compare different method to reconstruct the density matrix from the wigner function.
    """
    dir_name = "output"
    N = 50
    inter_sizes = [10, 20, 30, 50]

    # Check if the directory exists, and create if it doesn't
    if not os.path.exists(dir_name):
        print("Create output directory...")
        os.makedirs(dir_name)

    pre_computation(inter_sizes, N)

    try:
        print(f"1. Test 3 reconstruction methods on Synth data, using interpolation to reduce the wigner input size {inter_sizes}")
        synth_dir = dir_name + "/sinthetic/"
        print("create dir for synth data")
        if not os.path.exists(synth_dir):
            print("Create output directory...")
            os.makedirs(synth_dir)
        time_vs_sizes_1 = {}
        fidelity_vs_sizes_1 = {}
        for n, fname, m in METHOD:
            try:
                print("------------")
                print(f"Try method {n}")
                data  = test_w_synth_file(m, inter_sizes, N)
                size, ex_time, size2, fidelity = process_data_for_synth_bm(data, inter_sizes, synth_dir, fname)
                time_vs_sizes_1[n] = (size, ex_time)
                fidelity_vs_sizes_1[n] = (size2, fidelity)
                plot_multiple_2d_lines(time_vs_sizes_1, synth_dir+"compare_time_agv_vs_size", title="Avg time vs Size", x_label="size", y_label="time")
                plot_multiple_2d_lines(fidelity_vs_sizes_1, synth_dir+"compare_fidel_agv_vs_size", title="Avg fidelity vs Size", x_label="size", y_label="fidelity")
            except Exception as e:
                print(f"Something went wrong: {e}")
    except Exception as e:
        print(f"Something went wrong: {e}")        

    try:
        print(f"2. Test 3 Method with know input perturbed by controlled gaussian noise")
        gauss_noise_level = [0, 0.01, 0.02, 0.04]
        time_vs_sizes_2 = {}
        fidelity_vs_sizes_2 = {}
        for n, fname, m in METHOD:
            data = test_w_qstate(m, inter_sizes, gauss_noise_level, N)
            # time_vs_sizes[n] = (size, ex_time)
            # fidelity_vs_sizes[n] = (size2, fidelity)
        
    # Use time_vs_sizes and 

    except Exception as e:
        print(f"Something went wrong: {e}")     

benchmark()