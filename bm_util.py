import pickle
import matplotlib.pyplot as plt
import numpy as np
from qutip import fidelity, Qobj
import os
from scipy.interpolate import RegularGridInterpolator
import time
import sven_solution as ss
import numpy as np
import matplotlib.pyplot as plt
import max_solution as ms
import niko_solution as ns
import sven_solution as ss

SYNTHETIC_DATA = ['noisy_wigner_' + str(i) + '.pickle' for i in range(16)]

SYNTHETIC_EXPECTED_OUTCOME = ['quantum_state_' + str(i) + '.pickle' for i in range(8)]

def plot_multiple_2d_lines(data_dict, filename, title="2D Lines Plot", x_label="X", y_label="Y"):
    plt.figure(figsize=(8, 6))

    for label, (x_list, y_list) in data_dict.items():
        if not x_list or not y_list:
            continue
        try:
            # Optional: sort by x if needed
            sorted_points = sorted(zip(x_list, y_list))
            x_vals, y_vals = zip(*sorted_points)
            plt.plot(x_vals, y_vals, marker='o', label=label)
        except ValueError as e:
            print(f"Something went wrong with {label}: {e}")

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def add_wigner_noise(W, sigma=0.01, seed=None):
    """
    Add Gaussian noise to a Wigner function.

    Parameters:
        W (np.ndarray): 2D array representing the Wigner function W(x, p).
        sigma (float): Standard deviation of the Gaussian noise.
        seed (int or None): Optional seed for reproducibility.

    Returns:
        W_noisy (np.ndarray): Noisy Wigner function.
    """
    if seed is not None:
        np.random.seed(seed)

    noise = np.random.normal(loc=0.0, scale=sigma, size=W.shape)
    return W + noise

def load_synth_file(filename):
    path='data/synthetic/' + filename
    with open(path, 'rb') as f:
        x, y, w = pickle.load(f)
    return x,y,w

def load_synth_outcome(filename):
    path='data/synthetic/' + filename
    with open(path, 'rb') as f:
        m = pickle.load(f)
    return m

def interpolate_to_small_grid(x, y, z, new_size=50):
    x_new = np.linspace(min(x), max(x), new_size)
    y_new = np.linspace(min(y), max(y), new_size)
    interpolator = RegularGridInterpolator((y, x), z)
    Xn, Yn = np.meshgrid(x_new, y_new)
    points = np.stack([Yn.ravel(), Xn.ravel()], axis=-1)
    z_interp = interpolator(points).reshape((new_size, new_size))
    return x_new, y_new, z_interp


def test_w_synth_file(func, sizes, N):
    data = {}
    for i in range(8):
        try:
            print(f"Load data from noisy_wigner_{i}")
            x, y, z = load_synth_file(SYNTHETIC_DATA[i])
            expected_rho = load_synth_outcome(SYNTHETIC_EXPECTED_OUTCOME[i])
        except Exception as e:
            print(f"Something went wrong: {e}")

        try:
            exper_rho = []
            for kk in range(len(sizes)):
                rs = sizes[kk]
                try:
                    print(f"Compute interpolation")
                    nx, ny, nw = pre_process_input(x, y, z, N, interpolation=rs)
                    print(f"Start reconstruction of noisy_wigner_{i}, with size {rs}")
                    start_time = time.time()
                    out = func(nx, ny, nw, N)
                    end_time = time.time()
                    exec_time = end_time - start_time
                    exper_rho.append((rs, out, expected_rho, fidelity(Qobj(out, dims=[[N],[N]]), Qobj(expected_rho, dims=[[N],[N]])), exec_time))
                    print(f"Reconstruction complete in {exec_time:2f}s")
                except Exception as e:
                    print(f"Something went wrong: {e}")

            data[i] = exper_rho
        except Exception as e:
            print(f"Something went wrong: {e}")
    return data

def process_data_for_synth_bm(data, sizes, filename, method_name):
    # Set up the plot grid (e.g., 2 rows x 4 columns)
    num_keys = len(data)
    fig, axes = plt.subplots(nrows=3, ncols=num_keys, figsize=(5 * num_keys, 12))

    # If only one column, axes[:, i] indexing won't work directly
    if num_keys == 1:
        axes = np.expand_dims(axes, axis=1)

    exec_time_vs_size = {key: [] for key in sizes}
    fidelity_vs_size = {key: [] for key in sizes}
    try:
        print("Try to Plot the result of synth benchmark")
        for col_idx, key in enumerate(data):
            try:
                content = data[key]
                s_values, o_values, eo_values, f_values, t_values = zip(*content)
                
                # Convert to lists
                s_values = list(s_values)
                f_values = list(f_values)
                t_values = list(t_values)

                for s, t in zip(s_values, t_values):
                    if s in exec_time_vs_size:
                        exec_time_vs_size[s].append(t)

                for s, f in zip(s_values, f_values):
                    if s in fidelity_vs_size:
                        fidelity_vs_size[s].append(f)
                # Assume o_values and eo_values are 2D numpy arrays (matrices)
                # Stack into one array if needed
                o_mat = np.array(o_values)
                eo_mat = np.array(eo_values)
                for s in range(len(s_values)):
                    print(f"noisy_wigner_{key} : size {s_values[s]}, fidelity {f_values[s]}, exec time {t_values[s]}")

                # Row 0: Heatmap of o_values
                ax_o = axes[0, col_idx]
                im_o = ax_o.imshow(np.real(o_mat[-1]), aspect='auto', cmap='viridis')
                ax_o.set_title(f"{key} - o_values")
                fig.colorbar(im_o, ax=ax_o)
                # Row 1: Heatmap of eo_values
                ax_eo = axes[1, col_idx]
                im_eo = ax_eo.imshow(np.real(eo_mat[-1]), aspect='auto', cmap='plasma')
                ax_eo.set_title(f"{key} - eo_values")
                fig.colorbar(im_eo, ax=ax_eo)

                # Row 2: Line plot of f_values vs t_values
                ax_f_t = axes[2, col_idx]
                ax_f_t.plot(s_values, f_values, marker='o')
                ax_f_t.set_xlabel("s_values")
                ax_f_t.set_ylabel("f_values")
                ax_f_t.set_title(f"{key} - f vs s")
            except Exception as e:
                print(f"Something went wrong: {e}")
        # Overall layout adjustment
        plt.tight_layout()
        plt.savefig(filename + method_name+  ".png", dpi=300)
    except Exception as e:
        print(f"Something went wrong: {e}")
    # Plot exec time
    try:
        print("plot execution times")
        sorted_s = sorted(exec_time_vs_size.keys())
        sorted_avg_t = [np.mean(exec_time_vs_size[s]) for s in sorted_s]

        
        plt.figure(figsize=(8, 5))
        plt.plot(sorted_s, sorted_avg_t, marker='o')
        plt.xlabel("size after interpolation")
        plt.ylabel("Average execution time")
        plt.title("Average exec time vs size")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(filename +method_name+ "_exTime.png", dpi=300)

        sorted_s = sorted(fidelity_vs_size.keys())
        sorted_avg_fid = [np.mean(fidelity_vs_size[s]) for s in sorted_s]

    except Exception as e:
        print(f"Something went wrong: {e}")
    return sorted_s, sorted_avg_t, sorted_s, sorted_avg_fid

def test_w_qstate(funct, inter_sizes, gauss_noise_level, N):
    """_summary_

    Args:
        m (_type_): funtion to SOLVE
        inter_sizes (_type_): _description_
        gauss_noise_level (_type_): _description_
        N (_type_): _description_
        add_arg (_type_, optional): _description_. Defaults to synt_precomp_list.

    Returns:
        _type_: _description_
    """
    return None

def pre_computation(inter_sizes, N):
    #Sven precomputation
    try:
        print("Do some precomputation for Method Sven")
        start = time.time()
        for si in inter_sizes:
            x_lim = 6
            y_lim = 6
            N_x = si
            N_y = si
            xvec = np.linspace(-x_lim, x_lim, N_x)
            yvec = np.linspace(-y_lim, y_lim, N_y)
            fname=  f"svendata__N{N}_x{x_lim}_{N_x}_y{y_lim}_{N_y}.npy"
            data = []
            if os.path.exists(fname):
                print("data file exist already computed")
                try:
                    data = np.load("precomputation/"+fname)
                except Exception as e:
                    data = []
            if data.size==0:
                data = ss.precomp(N, xvec, yvec)
                np.save("precomputation/"+fname, data)
        end = time.time()
        print(f"Precomputation for size {inter_sizes} took {end-start} seconds")
        print("-----------")    
    except Exception as e:
        print(f"Something went wrong: {e}") 
    
    #Max precomputation
    try:
        print("Do some precomputation for Max Method")
        # Setup parameters

        start = time.time()
        for si in inter_sizes:
            x_lim = 6
            y_lim = 6
            N_x = si
            N_y = si
            xvec = np.linspace(-x_lim, x_lim, N_x)
            yvec = np.linspace(-y_lim, y_lim, N_y)
            X, Y = np.meshgrid(xvec, yvec)
            path = ms.get_precomputed_path(N_x, N_y, N)
            if os.path.exists(path):
                print(f"cached basis exist from {path}")
            else:
                data = ms.precompute_W_basis(xvec, yvec, N)
                np.save("precomputation/"+path, data)
                print(f"Saved Wigner basis to {path}")
        end = time.time()
        print(f"Precomputation for size {inter_sizes} took {end-start} seconds")
        print("-----------")
    except Exception as e:
        print(f"Something went wrong: {e}")

    # Least square precomputation
    try:
        print("Do some precomputation for Least square")
        # Setup parameters
        start = time.time()
        for si in inter_sizes:
            x_lim = 6
            y_lim = 6
            N_x = si
            N_y = si
            xvec = np.linspace(-x_lim, x_lim, N_x)
            yvec = np.linspace(-y_lim, y_lim, N_y)
            X, Y = np.meshgrid(xvec, yvec)
            alphas = (X + 1j * Y).ravel()
            fname=  f"E_ops_chunk_N{N}_x{x_lim}_{N_x}_y{y_lim}_{N_y}.npy"
            if os.path.exists(fname):
                print(f"Data for N={N} exists already")
                data = np.load("precomputation/"+fname)
            else:
                ns.construct_measurement_operators_joblib_chunked(
                    alphas, N, chunk_size=10000, N_buffer=0, save=True,
                    x_lim=x_lim, y_lim=y_lim, N_x=N_x, N_y=N_y
                )

        end = time.time()
        print(f"Precomputation for size {inter_sizes} took {end-start} seconds")
    except Exception as e:
        print(f"Something went wrong: {e}")

    print("-----------")

def proceed_data(data):
    return None

def pre_process_input(x, y, w, N, interpolation = 10, smooting=0.1):
    w = np.nan_to_num(w, nan=0.0)
    nx, ny, nw = interpolate_to_small_grid(x, y, w, interpolation)



    return nx, ny, nw