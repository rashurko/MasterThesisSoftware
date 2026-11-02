#!/usr/bin/env python3
r"""
run_projection.py: Performs 2-RDM representability projection methods based on
the algorithms in Hamilton_code.docx, using T and V matrices extracted from C++ JSON outputs.

Features:
- Extracts T, V, N, M, and g from exactSol.json (or user-specified JSON file).
- Allows overriding N, M, or g from the command-line.
- Computes the reduced Hamiltonian K and 4-index tensor h from T and V.
- Performs Dykstra projection, Halpern iteration, or Projected Gradient Descent.
- Validates representability: Tr(P), Tr(Q), Tr(G) and minimum eigenvalues.
- Compares calculated energy against exact FCI diagonalization.

Usage:
    python3 run_projection.py
    python3 run_projection.py --g 0.5 --method dykstra
    python3 run_projection.py --method halpern --max-iter 50
    python3 run_projection.py --method pgd --eta 0.05
    python3 run_projection.py --json exactSol.json --out result.json
"""

import argparse
from functools import partial
import json
import os
import sys
import numpy as np
from scipy.optimize import root
import scipy.sparse.linalg as spla


# =============================================================================
# 1. Tensor Reshaping and Positive Semidefinite Cone Projections
# =============================================================================

def tensor2matrix(gamma):
    """Reshapes rank-4 or rank-6 tensor into a 2D matrix."""
    shape = gamma.shape
    rank = gamma.ndim
    if rank == 6:
        n = shape[0]
        return np.reshape(gamma, (n**3, n**3)), n
    elif rank == 4:
        n = shape[0]
        return np.reshape(gamma, (n**2, n**2)), n
    else:
        raise ValueError("Only rank-4 and rank-6 tensors are supported.")


def matrix2tensor(matrix, n):
    """Reshapes 2D matrix back into rank-4 or rank-6 tensor."""
    shape = matrix.shape
    if shape == (n**2, n**2):
        return np.reshape(matrix, (n, n, n, n))
    elif shape == (n**3, n**3):
        return np.reshape(matrix, (n, n, n, n, n, n))
    else:
        raise ValueError(f"Matrix shape {shape} does not match n^2 ({n**2}) or n^3 ({n**3}).")


def calculate_shift(eigenvalues, alpha):
    r"""
    Calculates shift sigma0 so that sum_i max(0, lambda_i - sigma0) = alpha.
    Uses SciPy root finder with a reliable 60-iteration bisection fallback.
    """
    trace_fn = lambda s0: np.sum(np.heaviside(eigenvalues - s0, 0.5) * (eigenvalues - s0))
    constraint = lambda x: trace_fn(x) - alpha
    
    res = root(constraint, 0.0)
    if res.success:
        s0 = float(res.x[0]) if hasattr(res.x, '__len__') else float(res.x)
        if abs(trace_fn(s0) - alpha) < 1e-6:
            return s0

    # Bisection fallback
    low = float(np.min(eigenvalues) - alpha)
    high = float(np.max(eigenvalues))
    for _ in range(60):
        mid = 0.5 * (low + high)
        if trace_fn(mid) > alpha:
            low = mid
        else:
            high = mid
    return float(0.5 * (low + high))


def find_closest_sdp(gamma, trace):
    """Metric projection onto the positive semidefinite cone with fixed trace."""
    gamma_reshaped, n = tensor2matrix(gamma)
    L = 0.5 * (gamma_reshaped + gamma_reshaped.conj().T)
    vals, vecs = np.linalg.eigh(L)
    sigma0 = calculate_shift(vals, trace)
    shifted_eigvals = np.heaviside(vals - sigma0, 0.5) * (vals - sigma0)
    L_closest = vecs @ np.diag(shifted_eigvals) @ vecs.conj().T
    return matrix2tensor(L_closest, n)


# =============================================================================
# 2. Representability Mappings and Exact Adjoints
# =============================================================================

def project_antisymmetric(tensor):
    """Projects onto strict fermionic antisymmetric subspace."""
    return 0.25 * (
        tensor 
        - np.einsum('bacd -> abcd', tensor) 
        - np.einsum('abdc -> abcd', tensor) 
        + np.einsum('badc -> abcd', tensor)
    )


def calc_P(gamma, N_electrons=None, M_dim=None, conjugate=False, N=None, M=None):
    """P-condition mapping (identity)."""
    return gamma


def calc_Q(tensor, N_electrons=None, M_dim=None, conjugate=False, linear_only=False, N=None, M=None):
    """Q-condition mapping (affine, linear-only, or exact adjoint)."""
    if N_electrons is None:
        N_electrons = N
    if M_dim is None:
        M_dim = M if M is not None else tensor.shape[0]

    eye = np.eye(M_dim)
    scale = 1.0 / (N_electrons - 1.0)

    if not conjugate:
        traced_tensor = np.einsum('abgb -> ag', tensor) * scale
        wedge_I_rho = (
             np.einsum('ag, bd -> abgd', eye, traced_tensor)
            -np.einsum('ad, bg -> abgd', eye, traced_tensor)
            -np.einsum('bg, ad -> abgd', eye, traced_tensor)
            +np.einsum('bd, ag -> abgd', eye, traced_tensor)
        )
        if linear_only:
            return tensor - wedge_I_rho
        else:
            wedge_I_I = (np.einsum('ag, bd -> abgd', eye, eye) - np.einsum('ad, bg -> abgd', eye, eye))
            return tensor + wedge_I_I - wedge_I_rho
    else:
        K1 =  np.einsum('abgd, ag -> bd', tensor, eye) * scale
        K2 = -np.einsum('abgd, ad -> bg', tensor, eye) * scale
        K3 = -np.einsum('abgd, bg -> ad', tensor, eye) * scale
        K4 =  np.einsum('abgd, bd -> ag', tensor, eye) * scale
        K_total = K1 + K2 + K3 + K4
        wedge_I_rho_adj = np.einsum('bd, ag -> abgd', eye, K_total)
        return tensor - wedge_I_rho_adj


def calc_G(tensor, N_electrons=None, M_dim=None, conjugate=False, linear_only=False, N=None, M=None):
    """Self-adjoint G-condition mapping (particle-hole positivity)."""
    if N_electrons is None:
        N_electrons = N
    if M_dim is None:
        M_dim = M if M is not None else tensor.shape[0]

    eye = np.eye(M_dim)
    scale = 1.0 / (N_electrons - 1.0)
    traced_tensor = np.einsum('abgb -> ag', tensor) * scale
    term_1 = np.einsum('bd, ag -> abgd', eye, traced_tensor)
    term_2 = -np.einsum('adgb -> abgd', tensor)
    return term_1 + term_2


# =============================================================================
# 3. CG Normal Solver and Projection Step
# =============================================================================

def solve_lagrange_multiplier(apply_A, apply_A_dagger, Delta, tol=1e-6, maxiter=1000):
    """Solves (A A^dagger + 1e-8 I) lambda = Delta using matrix-free CG."""
    constraint_shape = Delta.shape
    vec_size = Delta.size

    def matvec(lambda_vec):
        lambda_tensor = lambda_vec.reshape(constraint_shape)
        param_tensor = apply_A_dagger(lambda_tensor)
        constraint_tensor_new = apply_A(param_tensor)
        regularized_output = constraint_tensor_new + 1e-8 * lambda_tensor
        return regularized_output.reshape(-1)

    M_operator = spla.LinearOperator(shape=(vec_size, vec_size), matvec=matvec, dtype=Delta.dtype)
    rhs_vec = Delta.reshape(-1)
    try:
        lambda_sol_vec, _ = spla.cg(M_operator, rhs_vec, tol=tol, maxiter=maxiter)
    except TypeError:
        lambda_sol_vec, _ = spla.cg(M_operator, rhs_vec, rtol=tol, maxiter=maxiter)
    return lambda_sol_vec.reshape(constraint_shape)


def apply_projection_step(tensor_current, forward_map_eval, forward_map_linear, adjoint_map, target_projector, cg_tol=1e-6, cg_maxiter=1000):
    """Computes constraint residual, solves dual normal equation, and updates Gamma."""
    mapped_current = forward_map_eval(tensor_current)
    target_tensor = target_projector(mapped_current)
    delta_constraint = target_tensor - mapped_current
    residual_norm = float(np.linalg.norm(delta_constraint))

    lmbda = solve_lagrange_multiplier(
        apply_A=forward_map_linear,
        apply_A_dagger=adjoint_map,
        Delta=delta_constraint,
        tol=cg_tol,
        maxiter=cg_maxiter
    )
    tensor_corrected = tensor_current + adjoint_map(lmbda)
    return tensor_corrected, residual_norm


# =============================================================================
# 4. Projection Algorithms (Dykstra, Halpern, and PGD)
# =============================================================================

def run_dykstra(tensor_initial, N, M, traces, h=None, max_iter=50, outer_tol=1e-5, cg_tol=1e-4, verbose=True):
    """Executes Dykstra's cyclic projection algorithm with separate memory increments."""
    constraints_list = [
        {
            'name': 'Antisymmetry',
            'forward_map_eval': lambda x: x,
            'forward_map_linear': lambda x: x,
            'adjoint_map': lambda x: x,
            'target_projector': project_antisymmetric
        },
        {
            'name': 'P-Condition',
            'forward_map_eval': lambda x: x,
            'forward_map_linear': lambda x: x,
            'adjoint_map': lambda x: x,
            'target_projector': partial(find_closest_sdp, trace=traces['P'])
        },
        {
            'name': 'Q-Condition',
            'forward_map_eval': partial(calc_Q, N_electrons=N, M_dim=M, conjugate=False, linear_only=False),
            'forward_map_linear': partial(calc_Q, N_electrons=N, M_dim=M, conjugate=False, linear_only=True),
            'adjoint_map': partial(calc_Q, N_electrons=N, M_dim=M, conjugate=True),
            'target_projector': partial(find_closest_sdp, trace=traces['Q'])
        },
        {
            'name': 'G-Condition',
            'forward_map_eval': partial(calc_G, N_electrons=N, M_dim=M, conjugate=False),
            'forward_map_linear': partial(calc_G, N_electrons=N, M_dim=M, conjugate=False),
            'adjoint_map': partial(calc_G, N_electrons=N, M_dim=M, conjugate=True),
            'target_projector': partial(find_closest_sdp, trace=traces['G'])
        }
    ]

    tensor_current = tensor_initial.copy()
    increments = [np.zeros_like(tensor_initial) for _ in constraints_list]

    for iteration in range(max_iter):
        max_res = 0.0
        tensor_old = tensor_current.copy()

        for c_idx, constraint in enumerate(constraints_list):
            pre_state = tensor_current - increments[c_idx]
            tensor_projected, res_norm = apply_projection_step(
                tensor_current=pre_state,
                forward_map_eval=constraint['forward_map_eval'],
                forward_map_linear=constraint['forward_map_linear'],
                adjoint_map=constraint['adjoint_map'],
                target_projector=constraint['target_projector'],
                cg_tol=cg_tol,
                cg_maxiter=500
            )
            max_res = max(max_res, res_norm)
            increments[c_idx] = tensor_projected - pre_state
            tensor_current = tensor_projected

        diff_norm = float(np.linalg.norm(tensor_current - tensor_old))
        if verbose and (iteration % 5 == 0 or iteration == 0):
            e_str = f", Energy: {np.sum(tensor_current * h):.6f}" if h is not None else ""
            print(f"  [Dykstra Iter {iteration:3d}] Diff Norm: {diff_norm:.6e}, Max Res: {max_res:.6e}{e_str}")

        if diff_norm < outer_tol and iteration > 0:
            if verbose:
                print(f"  Dykstra converged in {iteration + 1} iterations.")
            break
    return tensor_current


def run_halpern(tensor_initial, N, M, traces, h=None, max_iter=50, outer_tol=1e-5, cg_tol=1e-4, verbose=True):
    """Executes Halpern fixed-point iteration: D_{n+1} = 1/(n+1) Anchor + n/(n+1) T(D_n)."""
    constraints_list = [
        {
            'name': 'Antisymmetry',
            'forward_map_eval': lambda x: x,
            'forward_map_linear': lambda x: x,
            'adjoint_map': lambda x: x,
            'target_projector': project_antisymmetric
        },
        {
            'name': 'P-Condition',
            'forward_map_eval': lambda x: x,
            'forward_map_linear': lambda x: x,
            'adjoint_map': lambda x: x,
            'target_projector': partial(find_closest_sdp, trace=traces['P'])
        },
        {
            'name': 'Q-Condition',
            'forward_map_eval': partial(calc_Q, N_electrons=N, M_dim=M, conjugate=False, linear_only=False),
            'forward_map_linear': partial(calc_Q, N_electrons=N, M_dim=M, conjugate=False, linear_only=True),
            'adjoint_map': partial(calc_Q, N_electrons=N, M_dim=M, conjugate=True),
            'target_projector': partial(find_closest_sdp, trace=traces['Q'])
        },
        {
            'name': 'G-Condition',
            'forward_map_eval': partial(calc_G, N_electrons=N, M_dim=M, conjugate=False),
            'forward_map_linear': partial(calc_G, N_electrons=N, M_dim=M, conjugate=False),
            'adjoint_map': partial(calc_G, N_electrons=N_electrons, M_dim=M, conjugate=True),
            'target_projector': partial(find_closest_sdp, trace=traces['G'])
        }
    ]

    anchor = tensor_initial.copy()
    tensor_current = tensor_initial.copy()

    for iteration in range(max_iter):
        j = iteration + 1
        gamma_tmp = tensor_current
        max_res = 0.0

        for constraint in constraints_list:
            tensor_projected, res_norm = apply_projection_step(
                tensor_current=gamma_tmp,
                forward_map_eval=constraint['forward_map_eval'],
                forward_map_linear=constraint['forward_map_linear'],
                adjoint_map=constraint['adjoint_map'],
                target_projector=constraint['target_projector'],
                cg_tol=cg_tol,
                cg_maxiter=500
            )
            max_res = max(max_res, res_norm)
            gamma_tmp = tensor_projected

        tensor_current = (1.0 / (j + 1)) * anchor + (j / (j + 1)) * gamma_tmp
        if verbose and (iteration % 5 == 0 or iteration == 0):
            e_str = f", Energy: {np.sum(tensor_current * h):.6f}" if h is not None else ""
            print(f"  [Halpern Iter {iteration:3d}] Max Cone Res: {max_res:.6e}{e_str}")

        if max_res < outer_tol and iteration > 0:
            if verbose:
                print(f"  Halpern converged in {iteration + 1} iterations.")
            break
    return tensor_current


def run_projected_gradient_descent(N, M, traces, h_tensor, eta0=0.05, max_iter=50, verbose=True):
    """
    Solves min <h, Gamma> s.t. Gamma in C via Projected Gradient Descent.
    Eliminates the Tikhonov penalty and outward normal ray lock!
    """
    # Start on P-cone
    Gamma = find_closest_sdp(np.zeros_like(h_tensor), trace=traces['P'])
    Gamma = project_antisymmetric(Gamma)

    for k in range(max_iter):
        eta = eta0 / np.sqrt(k + 1.0)
        # Gradient descent step
        Gamma_step = Gamma - eta * h_tensor

        # Projection back onto representable cone (5 cyclic steps)
        Gamma_proj = Gamma_step
        for _ in range(3):
            Gamma_proj = project_antisymmetric(Gamma_proj)
            Gamma_proj = find_closest_sdp(Gamma_proj, trace=traces['P'])
            # Q projection
            Q_affine = calc_Q(Gamma_proj, N_electrons=N, M_dim=M, conjugate=False, linear_only=False)
            Q_sdp = find_closest_sdp(Q_affine, trace=traces['Q'])
            dQ = solve_lagrange_multiplier(
                apply_A=partial(calc_Q, N_electrons=N, M_dim=M, conjugate=False, linear_only=True),
                apply_A_dagger=partial(calc_Q, N_electrons=N, M_dim=M, conjugate=True),
                Delta=Q_sdp - Q_affine,
                tol=1e-4, maxiter=200
            )
            Gamma_proj += calc_Q(dQ, N_electrons=N, M_dim=M, conjugate=True)

            # G projection
            G_curr = calc_G(Gamma_proj, N_electrons=N, M_dim=M, conjugate=False)
            G_sdp = find_closest_sdp(G_curr, trace=traces['G'])
            dG = solve_lagrange_multiplier(
                apply_A=partial(calc_G, N_electrons=N, M_dim=M, conjugate=False),
                apply_A_dagger=partial(calc_G, N_electrons=N, M_dim=M, conjugate=True),
                Delta=G_sdp - G_curr,
                tol=1e-4, maxiter=200
            )
            Gamma_proj += calc_G(dG, N_electrons=N, M_dim=M, conjugate=True)

        step_norm = float(np.linalg.norm(Gamma_proj - Gamma))
        Gamma = Gamma_proj
        energy = float(np.sum(h_tensor * Gamma))

        if verbose and (k % 5 == 0 or k == 0):
            print(f"  [PGD Iter {k:3d}] Energy: {energy:10.6f}, Step Norm: {step_norm:.6e}")

        if step_norm < 1e-4 and k > 5:
            if verbose:
                print(f"  PGD converged to SDP ground state in {k + 1} iterations.")
            break
    return Gamma


# =============================================================================
# 5. C++ Matrix Conversion and Hamiltonian Assembly
# =============================================================================

def get_idx_v(i, j, M):
    """Computes ordered pair index matching C++ getIdxV(i, j)."""
    if i < j:
        return (j - i - 1) + (i * (2 * M - i - 1)) // 2, 1
    elif i > j:
        return (i - j - 1) + (j * (2 * M - j - 1)) // 2, -1
    else:
        return 0, 0


def build_k_matrix(T, V, N, M, g):
    """Computes reduced Hamiltonian matrix K from T and V matching C++ generateKMatrix()."""
    P = M * (M - 1) // 2
    K = g * V.copy()
    for j in range(M):
        for i in range(j):
            idx1, s1 = get_idx_v(i, j, M)
            for l in range(M):
                for k in range(l):
                    idx2, s2 = get_idx_v(k, l, M)
                    factorT = 0.0
                    if j == l: factorT += T[i, k]
                    if i == k: factorT += T[j, l]
                    if i == l: factorT -= T[j, k]
                    if j == k: factorT -= T[i, l]
                    K[idx1, idx2] += (1.0 / (N - 1.0)) * s1 * s2 * factorT
    return K


def k_matrix_to_tensor(K_mat, M):
    """Converts P x P K-matrix to 4-index tensor with fermionic signs."""
    K_tensor = np.zeros((M, M, M, M), dtype=np.float64)
    for i in range(M):
        for j in range(M):
            idx1, s1 = get_idx_v(i, j, M)
            if s1 == 0: continue
            for k in range(M):
                for l in range(M):
                    idx2, s2 = get_idx_v(k, l, M)
                    if s2 == 0: continue
                    K_tensor[i, j, k, l] = s1 * s2 * K_mat[idx1, idx2]
    return K_tensor


def tensor_to_pair_matrix(tensor, M):
    """Extracts P x P matrix from (M, M, M, M) antisymmetric tensor."""
    P = M * (M - 1) // 2
    mat = np.zeros((P, P), dtype=np.float64)
    for j in range(M):
        for i in range(j):
            idx1, _ = get_idx_v(i, j, M)
            for l in range(M):
                for k in range(l):
                    idx2, _ = get_idx_v(k, l, M)
                    mat[idx1, idx2] = tensor[i, j, k, l]
    return mat


def check_representability(G_tensor, N, M):
    """Evaluates P, Q, and G minimum eigenvalues and traces."""
    P_mat, _ = tensor2matrix(G_tensor)
    Q_tensor = calc_Q(G_tensor, N_electrons=N, M_dim=M)
    Q_mat, _ = tensor2matrix(Q_tensor)
    G_cond_tensor = calc_G(G_tensor, N_electrons=N, M_dim=M)
    G_mat, _ = tensor2matrix(G_cond_tensor)

    eig_P = np.linalg.eigvalsh(0.5 * (P_mat + P_mat.T))
    eig_Q = np.linalg.eigvalsh(0.5 * (Q_mat + Q_mat.T))
    eig_G = np.linalg.eigvalsh(0.5 * (G_mat + G_mat.T))

    return {
        'tr_P': float(np.trace(P_mat)),
        'min_eig_P': float(np.min(eig_P)),
        'tr_Q': float(np.trace(Q_mat)),
        'min_eig_Q': float(np.min(eig_Q)),
        'tr_G': float(np.trace(G_mat)),
        'min_eig_G': float(np.min(eig_G)),
    }


# =============================================================================
# 6. Main Execution Pipeline
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Perform 2-RDM projection methods using T and V from C++ JSON.")
    parser.add_argument("--json", type=str, default="exactSol.json", help="Path to JSON file containing T, V, N, M (default: exactSol.json)")
    parser.add_argument("--N", type=int, default=None, help="Override number of fermions N")
    parser.add_argument("--M", type=int, default=None, help="Override number of single-particle states M")
    parser.add_argument("--g", type=float, default=None, help="Override interaction strength g")
    parser.add_argument("--method", type=str, default="dykstra", choices=["dykstra", "halpern", "pgd"], help="Algorithm: dykstra, halpern, or pgd (default: dykstra)")
    parser.add_argument("--alpha", type=float, default=1e-3, help="Step/regularization parameter alpha for Dykstra (default: 1e-3)")
    parser.add_argument("--eta", type=float, default=0.05, help="Step size eta for Projected Gradient Descent (default: 0.05)")
    parser.add_argument("--max-iter", type=int, default=30, help="Maximum outer iterations (default: 30)")
    parser.add_argument("--out", type=str, default=None, help="Optional output JSON file to save results")
    args = parser.parse_args()

    print("=" * 80)
    print(" 2-RDM PROJECTION METHOD RUNNER (Python implementation of Hamilton_code.docx)")
    print("=" * 80)

    # 1. Load C++ JSON file
    if not os.path.exists(args.json):
        print(f"Error: JSON file '{args.json}' not found.")
        sys.exit(1)

    with open(args.json) as f:
        data = json.load(f)

    # Extract or override parameters
    N = args.N if args.N is not None else data.get('N')
    M = args.M if args.M is not None else data.get('M')
    g = args.g if args.g is not None else data.get('g', 1.0)
    E_exact = data.get('eigenenergies', [None])[0] if 'eigenenergies' in data else None

    if 'T' not in data or 'V' not in data:
        print(f"Error: '{args.json}' must contain 'T' and 'V' matrices.")
        sys.exit(1)

    T_mat = np.array(data['T'], dtype=np.float64)
    V_mat = np.array(data['V'], dtype=np.float64)

    print(f"System Configuration:")
    print(f"  Fermions (N):                {N}")
    print(f"  Single-particle states (M):  {M}")
    print(f"  Interaction strength (g):    {g}")
    if E_exact is not None:
        print(f"  Exact FCI Energy in JSON:    {E_exact:.6f}")

    # 2. Reconstruct reduced Hamiltonian K matrix
    print("\n[Step 1] Constructing reduced Hamiltonian K matrix from T and V...")
    K_mat = build_k_matrix(T=T_mat, V=V_mat, N=N, M=M, g=g)
    K_tensor = k_matrix_to_tensor(K_mat, M)
    h_tensor = 0.25 * K_tensor  # 1/4 factor maps 4-index tensor to pair matrix

    target_traces = {
        'P': float(N * (N - 1)),
        'Q': float((M - N) * (M - N - 1)),
        'G': float(N * (M - N + 1))
    }
    print(f"  Target traces: Tr(P) = {target_traces['P']}, Tr(Q) = {target_traces['Q']}, Tr(G) = {target_traces['G']}")

    # 3. Execute Selected Method
    print(f"\n[Step 2] Running Projection Algorithm: '{args.method.upper()}' (max {args.max_iter} iters)...")
    if args.method == "dykstra":
        D0 = -(1.0 / args.alpha) * h_tensor
        Gamma_res = run_dykstra(
            tensor_initial=D0,
            N=N, M=M,
            traces=target_traces,
            h=h_tensor,
            max_iter=args.max_iter,
            outer_tol=1e-5,
            cg_tol=1e-4,
            verbose=True
        )
    elif args.method == "halpern":
        D0 = find_closest_sdp(np.zeros_like(h_tensor), trace=target_traces['P'])
        Gamma_res = run_halpern(
            tensor_initial=D0,
            N=N, M=M,
            traces=target_traces,
            h=h_tensor,
            max_iter=args.max_iter,
            outer_tol=1e-5,
            cg_tol=1e-4,
            verbose=True
        )
    elif args.method == "pgd":
        Gamma_res = run_projected_gradient_descent(
            N=N, M=M,
            traces=target_traces,
            h_tensor=h_tensor,
            eta0=args.eta,
            max_iter=args.max_iter,
            verbose=True
        )

    # 4. Representability and Energy Analysis
    print("\n[Step 3] Analyzing Representability of Projected 2-RDM...")
    stats = check_representability(Gamma_res, N, M)
    energy_computed = float(np.sum(h_tensor * Gamma_res))
    Gamma_pair_mat = tensor_to_pair_matrix(Gamma_res, M)

    print("\n" + "=" * 80)
    print(" FINAL RESULTS SUMMARY")
    print("=" * 80)
    print(f"  Computed Energy:         {energy_computed:12.6f}")
    if E_exact is not None:
        delta = energy_computed - E_exact
        print(f"  Exact FCI Energy:        {E_exact:12.6f}")
        print(f"  Energy Difference (ΔE):  {delta:+12.6f}")
    print(f"  P-trace (Gamma):         {stats['tr_P']:12.6f}  (Target: {target_traces['P']:6.1f}) | min eig: {stats['min_eig_P']:10.2e}")
    print(f"  Q-trace (2-hole):        {stats['tr_Q']:12.6f}  (Target: {target_traces['Q']:6.1f}) | min eig: {stats['min_eig_Q']:10.2e}")
    print(f"  G-trace (particle-hole): {stats['tr_G']:12.6f}  (Target: {target_traces['G']:6.1f}) | min eig: {stats['min_eig_G']:10.2e}")
    print("=" * 80)

    # 5. Save Output JSON if requested
    if args.out:
        out_data = {
            'N': N,
            'M': M,
            'g': g,
            'method': args.method,
            'energy': energy_computed,
            'E_exact': E_exact,
            'representability': stats,
            'Gamma_matrix': Gamma_pair_mat.tolist()
        }
        with open(args.out, 'w') as f:
            json.dump(out_data, f, indent=2)
        print(f"Saved results to: {args.out}")


if __name__ == "__main__":
    main()

