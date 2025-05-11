import numpy as np
import cvxpy as cp
import numpy as np
from scipy.special import comb, hermite
from itertools import product
from qutip import Qobj, wigner, basis
from qutip import fidelity
from qutip import coherent
import matplotlib.pyplot as plt
from tqdm import tqdm

# Re-run after code reset
import numpy as np
from qutip import displace, qeye, Qobj
from joblib import Parallel, delayed
from tqdm import tqdm

def ls_reconstruct(xvec, yvec, W, N):
    global E_ops

    # === Step 1: Normalize Wigner → probabilities w_k ===
    dx = xvec[1] - xvec[0]
    dy = yvec[1] - yvec[0]
    b = np.nanmean(W[0, :])  # crude background estimate at top edge
    a = 1 / (np.nansum(W - b) * dx * dy)  # normalization constant

    W_physical = a * (W - b)
    W_physical = np.nan_to_num(W_physical, nan=0.0)
    w_k = 0.5 * (1 + (np.pi / 2) * W_physical)
    w_k = w_k.ravel()

    # === Step 2: Create α_k grid (from x + ip) ===
    X, Y = np.meshgrid(xvec, yvec)
    alphas = (X + 1j * Y).ravel()
    x_lim = int(np.max(xvec))
    y_lim = int(np.max(yvec))
    N_x = len(xvec)
    N_y = len(yvec)
    fname=  f"E_ops_chunk_N{N}_x{x_lim}_{N_x}_y{y_lim}_{N_y}.npy"
    E_ops = np.load("precomputation/"+fname)
    E_ops = E_ops[:, :N, :N]#Qobj(E_ops[:N, :N], dims=[[N], [N]])

    safe_mask = np.abs(alphas) < 3

    def reconstruct_density_matrix(E_ops, w_k, N):
        """
        Solve the SDP to reconstruct the density matrix ρ.

        Inputs:
            E_ops: list of Qobj measurement operators E_k
            w_k: measurement probabilities
            N: Hilbert space dimension

        Returns:
            rho_opt (np.ndarray): estimated density matrix (N x N)
        """
        rho = cp.Variable((N, N), hermitian=True)

        constraints = [
            rho >> 0,                 # positive semidefinite
            cp.trace(rho) == 1        # normalized
        ]

        residuals = []
        for i, (E, w) in enumerate(zip(E_ops, w_k)):
            if not safe_mask[i]:
                continue  # skip unsafe regions
            E_np = E#.full()
            predicted = cp.real(cp.trace(E_np @ rho))
            residuals.append(predicted - w)

        loss = cp.sum_squares(cp.hstack(residuals))
        problem = cp.Problem(cp.Minimize(loss), constraints)
        problem.solve(solver=cp.SCS, verbose=True)

        print("     Optimization status:", problem.status)
        print("     Final loss:", problem.value)

        return rho.value

    rho_est = reconstruct_density_matrix(E_ops, w_k, N)

    return rho_est


def construct_E_op(alpha, N, N_large, P_large_data):
    P_large = Qobj(P_large_data, dims=[[N_large], [N_large]])
    D_large = displace(N_large, alpha)
    E_large = 0.5 * (qeye(N_large) + D_large * P_large * D_large.dag())
    E_small = Qobj(E_large.full()[:N, :N], dims=[[N], [N]])
    return E_small

def construct_measurement_operators_joblib_chunked(alphas, N, chunk_size=100000,
                                                    N_buffer=20, save=True,
                                                    x_lim=6, y_lim=6, N_x=1000, N_y=1000):
    N_large = N + N_buffer
    P_large_data = np.diag([(-1)**n for n in range(N_large)])

    num_chunks = 1

    for i in range(num_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, len(alphas))
        alphas_chunk = alphas[start:end]

        E_ops_chunk = Parallel(n_jobs=-1, backend="loky")(
            delayed(construct_E_op)(alpha, N, N_large, P_large_data)
            for alpha in tqdm(alphas_chunk, desc=f"Chunk {i+1}/{num_chunks}")
        )

        if save:
            E_dense = np.stack([E.full() for E in E_ops_chunk])
            filename = f"E_ops_chunk_N{N}_x{x_lim}_{N_x}_y{y_lim}_{N_y}.npy"
            np.save("precomputation/"+filename, E_dense)
            print(f"    Saved file {filename}")
