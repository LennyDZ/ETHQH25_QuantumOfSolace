# Re-run setup after code reset
import numpy as np
import os
import math
from scipy.special import comb, hermite
from itertools import product
from qutip import Qobj, wigner, basis
from qutip import fidelity
from qutip import coherent
import matplotlib.pyplot as plt
from tqdm import tqdm

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm
import os
import math
from scipy.special import comb, hermite


def get_precomputed_path(x, y, N):
    return f"precomputed_W_basis_N{N}_X{x}_Y{y}.npy"


def precompute_W_basis(x_arr, y_arr, N):
    path = get_precomputed_path(len(x_arr), len(y_arr), N)
    if os.path.exists(path):
        print(f"    Loading cached basis from {path}")
        return np.load("precomputation/"+path)
    
    print("     Generating new Wigner basis...")
    exp_term = np.exp(-2 * (x_arr[:, None] ** 2 + y_arr[None, :] ** 2)).astype(np.float64)

    pow_2 = {}
    norm_factors = {}
    combs = {}
    for n in tqdm(range(N), desc="Combinations"):
        combs[n] = [comb(n, k, exact=True) for k in range(n + 1)]

    for i in range(N):
        for j in range(N):
            pow_2[(i, j)] = 2 ** (i + j)
            norm_factors[(i, j)] = 1.0 / (math.sqrt(math.factorial(i)) * math.sqrt(math.factorial(j)))

    def single_W_ij(i, j):
        W_max = np.zeros((len(x_arr), len(y_arr)), dtype=complex)
        for k in range(i + 1):
            for l in range(j + 1):
                h1_vals = hermite(k + l)(2 * x_arr)
                h2_vals = hermite(i + j - k - l)(2 * y_arr)
                coef = combs[i][k] * combs[j][l] * (-1) ** l * (1j) ** (k + l)
                term = coef * np.outer(h1_vals, h2_vals)
                W_max += term
        scalar = complex((2 / np.pi) * norm_factors[(i, j)] / pow_2[(i, j)])
        return (scalar * exp_term * W_max).real

    W_max_arr = np.zeros((N, N, len(x_arr), len(y_arr)))
    for i in tqdm(range(N)):
        for j in range(i, N):
            Wij = single_W_ij(i, j)
            W_max_arr[i, j] = Wij
            if i != j:
                W_max_arr[j, i] = Wij
    return W_max_arr


def reconstruct_density_matrix_robust(x_small, y_small,W_small, N, add_arg = None):
    """
    Full pipeline: smooth, interpolate, normalize, reconstruct.
    """

    grid_resize = len(x_small)

    # Step 4: load or compute basis
    W_basis = precompute_W_basis(x_small, y_small, N).reshape(N * N, grid_resize * grid_resize).T
    # # Step 5: reconstruct
    A = np.linalg.pinv(W_basis.T @ W_basis) @ W_basis.T
    rho_vec = A @ W_small.ravel()*2

    return rho_vec.reshape(N, N)

   