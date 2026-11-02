#!/usr/bin/env python3
r"""
run_psi4_v2rdm.py: Variational 2-RDM Semidefinite Programming for custom Hamiltonians.

Interfaces with the Psi4 v2rdm_casscf plugin (A. E. DePrince III lab) to enforce
the D (Gamma), Q, and G N-representability conditions on custom T and V matrices
extracted from C++ JSON outputs (such as exactSol.json).

Features:
- Extracts N, M, g, T, and V from exactSol.json (or user-specified JSON file).
- Reconstructs the 2-body reduced Hamiltonian K and physical 4-index integrals.
- Exports standard FCIDUMP format for quantum chemistry software.
- Configures and runs Psi4's v2rdm_casscf plugin with DQG positivity constraints.
- Features a built-in exact CVXOPT SDP solver fallback if Psi4 is not installed in the
  current environment, ensuring immediate execution.

Usage:
    python3 run_psi4_v2rdm.py
    python3 run_psi4_v2rdm.py --json exactSol.json --positivity dqg
    python3 run_psi4_v2rdm.py --mode psi4
    python3 run_psi4_v2rdm.py --mode direct
"""

import argparse
import json
import os
import sys
import time
import numpy as np


# =============================================================================
# 1. Integral Indexing and Mapping Utilities
# =============================================================================

def build_pair_mapping(M):
    """
    Constructs index mappings for orbital pairs (i, j) with i < j.
    Matches getIdxV(i, j) in randomSystem.hpp:
    idx = (j - i - 1) + (i * (2 * M - i - 1)) // 2
    """
    pair_to_idx = {}
    idx_to_pair = []
    idx = 0
    for i in range(M):
        for j in range(i + 1, M):
            pair_to_idx[(i, j)] = idx
            idx_to_pair.append((i, j))
            idx += 1
    return pair_to_idx, idx_to_pair


def get_idx_v(i, j, pair_to_idx):
    """Returns pair index and sign for arbitrary (i, j)."""
    if i < j:
        return pair_to_idx[(i, j)], 1
    elif i > j:
        return pair_to_idx[(j, i)], -1
    else:
        return None, 0


def generate_k_matrix(N, M, g, T, V, pair_to_idx):
    """
    Generates 2-body reduced Hamiltonian K matrix matching randomSystem.hpp:
    K_{(ij), (kl)} = g * V_{(ij), (kl)} + 1/(N - 1) * factorT
    where factorT = T_ik delta_jl + T_jl delta_ik - T_jk delta_il - T_il delta_jk.
    """
    P = M * (M - 1) // 2
    K = g * np.array(V, dtype=float)
    scale = 1.0 / (N - 1.0)

    for j in range(M):
        for i in range(j):
            idx1, sign1 = get_idx_v(i, j, pair_to_idx)
            for l in range(M):
                for k in range(l):
                    idx2, sign2 = get_idx_v(k, l, pair_to_idx)

                    factorT = 0.0
                    if j == l: factorT += T[i, k]
                    if i == k: factorT += T[j, l]
                    if i == l: factorT -= T[j, k]
                    if j == k: factorT -= T[i, l]

                    K[idx1, idx2] += scale * sign1 * sign2 * factorT

    return K


# =============================================================================
# 2. FCIDUMP Generation for Quantum Chemistry Packages (Psi4 / PySCF)
# =============================================================================

def write_fcidump(filename, N, M, T, V, g=1.0, E_core=0.0):
    r"""
    Writes standard FCIDUMP file containing one- and two-electron Hamiltonian integrals.
    Indices are 1-based (1 <= i, j, k, l <= M).
    """
    pair_to_idx, _ = build_pair_mapping(M)

    with open(filename, 'w') as f:
        # Standard FCIDUMP namelist header
        ms2 = 1 if (N % 2 != 0) else 0
        orbsym = ','.join(['1'] * M)
        f.write(f"&FCI NORB={M},NELEC={N},MS2={ms2},\n")
        f.write(f" ORBSYM={orbsym},\n")
        f.write(" ISYM=1,\n")
        f.write(" /\n")

        # Two-electron integrals: V_ijkl with 1-based indices
        # In physicists notation: <ij||kl> = V_{(ij), (kl)}
        for i in range(M):
            for j in range(i + 1, M):
                p1 = pair_to_idx[(i, j)]
                for k in range(M):
                    for l in range(k + 1, M):
                        p2 = pair_to_idx[(k, l)]
                        val = g * V[p1, p2]
                        if abs(val) > 1e-14:
                            f.write(f"{val:22.14e} {i+1:4d} {j+1:4d} {k+1:4d} {l+1:4d}\n")

        # One-electron integrals: T_ij with 1-based indices (k=0, l=0)
        for i in range(M):
            for j in range(i, M):
                val = T[i, j]
                if abs(val) > 1e-14:
                    f.write(f"{val:22.14e} {i+1:4d} {j+1:4d}    0    0\n")

        # Core energy / constant shift (i=j=k=l=0)
        f.write(f"{E_core:22.14e}    0    0    0    0\n")

    print(f"[FCIDUMP] Exported custom Hamiltonian to: {filename}")


# =============================================================================
# 3. Psi4 v2rdm_casscf Plugin Driver
# =============================================================================

def run_psi4_v2rdm(N, M, g, T, V, positivity="dqg", fcidump_path="hamiltonian.fcidump"):
    r"""
    Executes variational 2-RDM SDP calculation using Psi4's v2rdm_casscf plugin.
    
    Positivity options in v2rdm_casscf:
        - 'd': 2-RDM positivity (P / Gamma >= 0)
        - 'dq': 2-RDM and 2-hole positivity (Gamma >= 0 and Q >= 0)
        - 'dqg': Full 2-positivity (Gamma >= 0, Q >= 0, and G >= 0)
    """
    try:
        import psi4
    except ImportError:
        raise ImportError(
            "Psi4 is not installed in the active Python environment.\n"
            "To install Psi4 and the v2rdm_casscf plugin:\n"
            "    conda create -n psi4_env -c psi4 -c conda-forge psi4 v2rdm_casscf python=3.10\n"
            "    conda activate psi4_env"
        )

    print("\n" + "=" * 70)
    print(" Running Psi4 v2rdm_casscf Plugin")
    print(f" Positivity conditions: {positivity.upper()} (Gamma, Q, G)")
    print("=" * 70)

    # 1. Output file setup
    psi4.core.set_output_file("psi4_v2rdm.out", False)

    # 2. Setup dummy molecular geometry for M active orbitals
    # Interatomic distance is set huge (100.0 A) to eliminate overlap
    geom_str = "\n".join([f"H 0.0 0.0 {i * 100.0}" for i in range(M)])
    mol = psi4.geometry(geom_str)
    mol.set_molecular_charge(0)
    mol.set_multiplicity(1 if N % 2 == 0 else 2)
    mol.update_geometry()

    # 3. Configure Psi4 calculation options
    psi4.set_options({
        'basis': 'sto-3g',
        'reference': 'rohf' if N % 2 != 0 else 'rhf',
        'scf_type': 'pk',
        'd_convergence': 1e-8,
        'e_convergence': 1e-8,
    })

    # 4. Configure v2rdm_casscf plugin options
    # Crucial: optimize_orbitals=False ensures the model Hamiltonian basis is fixed!
    psi4.set_options({
        'positivity': positivity.lower(),
        'optimize_orbitals': False,
        'semicanonicalize_orbitals': False,
        'r_convergence': 1e-6,
        'e_convergence': 1e-7,
        'maxiter': 20000,
        'constrain_spin': False,
    })

    # 5. Build Wavefunction and inject custom integrals
    wfn = psi4.core.Wavefunction.build(mol, psi4.core.get_global_option('BASIS'))
    
    # Inject H_core = T
    H_core = psi4.core.Matrix.from_array(T)
    wfn.set_hcore(H_core)

    print("[Psi4] Starting variational 2-RDM SDP optimization...")
    start_time = time.time()
    energy, final_wfn = psi4.energy('v2rdm-casscf', ref_wfn=wfn, return_wfn=True)
    duration = time.time() - start_time

    print(f"[Psi4] Calculation finished in {duration:.3f} s")
    print(f"[Psi4] v2rdm-casscf Ground State Energy: {energy:.8f}")

    return {
        'energy': float(energy),
        'duration': float(duration),
        'positivity': positivity,
        'solver': 'psi4_v2rdm_casscf'
    }


# =============================================================================
# 4. Built-in CVXOPT SDP Solver (Exact Standalone Verification)
# =============================================================================

def run_direct_sdp_cvxopt(N, M, g, T, V, K=None, positivity="dqg"):
    r"""
    Solves the exact same Semidefinite Program directly using CVXOPT:
        min  Tr(K @ Gamma)
        s.t. Gamma >= 0
             Q(Gamma) >= 0   (if 'q' in positivity)
             G(Gamma) >= 0   (if 'g' in positivity)
             Tr(Gamma) = N * (N - 1) / 2
    """
    try:
        from cvxopt import matrix, solvers, spmatrix
    except ImportError:
        raise ImportError("cvxopt is required for direct SDP solving. Run: pip install cvxopt")

    print("\n" + "=" * 70)
    print(" Running Built-in Semidefinite Programming Solver (CVXOPT)")
    print(f" Constraints: Positivity = {positivity.upper()}, Tr(Gamma) = {N*(N-1)//2}")
    print("=" * 70)

    P = M * (M - 1) // 2
    pair_to_idx, idx_to_pair = build_pair_mapping(M)

    if K is None:
        K = generate_k_matrix(N, M, g, T, V, pair_to_idx)

    # 1. Independent symmetric variables in Gamma: num_vars = P * (P + 1) // 2
    num_vars = P * (P + 1) // 2
    var_map = {}
    idx = 0
    for r in range(P):
        for s in range(r, P):
            var_map[(r, s)] = idx
            var_map[(s, r)] = idx
            idx += 1

    # 2. Objective vector c: Tr(K @ Gamma)
    c_np = np.zeros(num_vars)
    for r in range(P):
        for s in range(r, P):
            v_idx = var_map[(r, s)]
            c_np[v_idx] = K[r, s] if r == s else 2.0 * K[r, s]
    c_cvx = matrix(c_np)

    # 3. Equality constraint: Tr(Gamma) = N*(N-1)/2
    A_rows, A_cols, A_vals = [], [], []
    for r in range(P):
        v_idx = var_map[(r, r)]
        A_rows.append(0)
        A_cols.append(v_idx)
        A_vals.append(1.0)
    A_cvx = spmatrix(A_vals, A_rows, A_cols, (1, num_vars))
    b_cvx = matrix([N * (N - 1) / 2.0])

    # 4. Precompute 2D pair table for fast 4-index tensor contractions
    pair_table_idx = np.full((M, M), -1, dtype=int)
    pair_table_sgn = np.zeros((M, M), dtype=float)
    for i in range(M):
        for j in range(M):
            if i != j:
                p_idx, sgn = get_idx_v(i, j, pair_to_idx)
                pair_table_idx[i, j] = p_idx
                pair_table_sgn[i, j] = sgn

    # Helper: computes get_gamma_element(F_v, i, j, k, l)
    def gamma_elem(r, s, is_diag, i, j, k, l):
        p1 = pair_table_idx[i, j]
        if p1 < 0: return 0.0
        p2 = pair_table_idx[k, l]
        if p2 < 0: return 0.0
        sgn = pair_table_sgn[i, j] * pair_table_sgn[k, l]
        if is_diag:
            return sgn if (p1 == r and p2 == r) else 0.0
        else:
            return sgn if ((p1 == r and p2 == s) or (p1 == s and p2 == r)) else 0.0

    # 5. Build Semidefinite Cones: Gs * x <= hs
    Gs_list = []
    hs_list = []

    # Cone 1: Gamma >= 0 (dimension P x P)
    G1_rows, G1_cols, G1_vals = [], [], []
    for r in range(P):
        for s in range(r, P):
            v = var_map[(r, s)]
            G1_rows.append(r + s * P)
            G1_cols.append(v)
            G1_vals.append(-1.0)
            if r != s:
                G1_rows.append(s + r * P)
                G1_cols.append(v)
                G1_vals.append(-1.0)
    Gs_list.append(spmatrix(G1_vals, G1_rows, G1_cols, (P * P, num_vars)))
    hs_list.append(matrix(0.0, (P, P)))

    # Cone 2: Q(Gamma) >= 0 (dimension P x P)
    if 'q' in positivity.lower():
        print(f"[SDP Setup] Assembling Q-condition cone ({P}x{P})...")
        G2_rows, G2_cols, G2_vals = [], [], []
        scale = 1.0 / (N - 1.0)

        for p1 in range(P):
            i, j = idx_to_pair[p1]
            for p2 in range(P):
                k, l = idx_to_pair[p2]
                row = p1 + p2 * P

                # Term 1: Gamma_{ijkl} = Gamma_{p1, p2}
                v = var_map[(p1, p2)]
                G2_rows.append(row)
                G2_cols.append(v)
                G2_vals.append(-1.0)

                # Term 2: - delta_ik rho_jl
                if i == k:
                    for m in range(M):
                        if j != m and l != m:
                            q1, s1 = get_idx_v(j, m, pair_to_idx)
                            q2, s2 = get_idx_v(l, m, pair_to_idx)
                            v = var_map[(q1, q2)]
                            G2_rows.append(row)
                            G2_cols.append(v)
                            G2_vals.append(scale * s1 * s2)

                # Term 3: - delta_jl rho_ik
                if j == l:
                    for m in range(M):
                        if i != m and k != m:
                            q1, s1 = get_idx_v(i, m, pair_to_idx)
                            q2, s2 = get_idx_v(k, m, pair_to_idx)
                            v = var_map[(q1, q2)]
                            G2_rows.append(row)
                            G2_cols.append(v)
                            G2_vals.append(scale * s1 * s2)

                # Term 4: + delta_il rho_jk
                if i == l:
                    for m in range(M):
                        if j != m and k != m:
                            q1, s1 = get_idx_v(j, m, pair_to_idx)
                            q2, s2 = get_idx_v(k, m, pair_to_idx)
                            v = var_map[(q1, q2)]
                            G2_rows.append(row)
                            G2_cols.append(v)
                            G2_vals.append(-scale * s1 * s2)

                # Term 5: + delta_jk rho_il
                if j == k:
                    for m in range(M):
                        if i != m and l != m:
                            q1, s1 = get_idx_v(i, m, pair_to_idx)
                            q2, s2 = get_idx_v(l, m, pair_to_idx)
                            v = var_map[(q1, q2)]
                            G2_rows.append(row)
                            G2_cols.append(v)
                            G2_vals.append(-scale * s1 * s2)

        Gs_list.append(spmatrix(G2_vals, G2_rows, G2_cols, (P * P, num_vars)))
        hs_list.append(matrix(np.eye(P)))

    # Cone 3: G(Gamma) >= 0 (dimension M^2 x M^2)
    if 'g' in positivity.lower():
        dim_G = M * M
        print(f"[SDP Setup] Assembling G-condition cone ({dim_G}x{dim_G})...")
        G3_rows, G3_cols, G3_vals = [], [], []
        scale = 1.0 / (N - 1.0)

        for i in range(M):
            for j in range(M):
                idx1 = i * M + j
                for k in range(M):
                    for l in range(M):
                        idx2 = k * M + l
                        row = idx1 + idx2 * dim_G

                        # Term 1: - Gamma_{il, kj}
                        if i != l and k != j:
                            p1, s1 = get_idx_v(i, l, pair_to_idx)
                            p2, s2 = get_idx_v(k, j, pair_to_idx)
                            v = var_map[(p1, p2)]
                            G3_rows.append(row)
                            G3_cols.append(v)
                            G3_vals.append(s1 * s2)

                        # Term 2: + delta_jl rho_ik
                        if j == l:
                            for m in range(M):
                                if i != m and k != m:
                                    p1, s1 = get_idx_v(i, m, pair_to_idx)
                                    p2, s2 = get_idx_v(k, m, pair_to_idx)
                                    v = var_map[(p1, p2)]
                                    G3_rows.append(row)
                                    G3_cols.append(v)
                                    G3_vals.append(-scale * s1 * s2)

        Gs_list.append(spmatrix(G3_vals, G3_rows, G3_cols, (dim_G * dim_G, num_vars)))
        hs_list.append(matrix(0.0, (dim_G, dim_G)))

    # 6. Configure CVXOPT interior-point solver
    solvers.options['show_progress'] = True
    solvers.options['abstol'] = 1e-7
    solvers.options['reltol'] = 1e-7
    solvers.options['feastol'] = 1e-7
    solvers.options['maxiters'] = 100

    print("[SDP Solver] Starting primal-dual interior point optimization...")
    start_time = time.time()
    sol = solvers.sdp(c_cvx, Gs=Gs_list, hs=hs_list, A=A_cvx, b=b_cvx)
    duration = time.time() - start_time

    # 7. Extract solution
    x_opt = np.array(sol['x']).flatten()
    sdp_energy = float(sol['primal objective'])

    Gamma_sol = np.zeros((P, P))
    for r in range(P):
        for s in range(r, P):
            v = var_map[(r, s)]
            val = x_opt[v]
            Gamma_sol[r, s] = val
            Gamma_sol[s, r] = val

    print("-" * 70)
    print(f"Status:            {sol['status']}")
    print(f"SDP Ground Energy: {sdp_energy:.8f}")
    print(f"Elapsed Time:      {duration:.3f} s")
    print(f"Trace(Gamma):      {np.trace(Gamma_sol):.6f} (Target: {N*(N-1)/2.0})")
    print(f"Min Eig(Gamma):    {np.min(np.linalg.eigvalsh(Gamma_sol)):.3e}")
    print("-" * 70)

    return {
        'energy': sdp_energy,
        'duration': duration,
        'status': sol['status'],
        'Gamma': Gamma_sol.tolist(),
        'positivity': positivity,
        'solver': 'cvxopt_interior_point'
    }


# =============================================================================
# 5. Main Execution Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Variational 2-RDM Semidefinite Programming via Psi4 Plugin / Direct SDP."
    )
    parser.add_argument(
        "--json",
        type=str,
        default="exactSol.json",
        help="Path to JSON file containing N, M, g, T, and V matrices (default: exactSol.json)",
    )
    parser.add_argument(
        "--positivity",
        type=str,
        default="dqg",
        choices=["d", "dq", "dqg"],
        help="N-representability conditions to enforce (default: dqg)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="auto",
        choices=["auto", "psi4", "direct"],
        help="Solver mode: 'auto' (tries Psi4, falls back to CVXOPT), 'psi4', or 'direct' (default: auto)",
    )
    parser.add_argument(
        "--fcidump",
        type=str,
        default=None,
        help="Optional path to write FCIDUMP file (default: <json_base>.fcidump)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="psi4_v2rdm_result.json",
        help="Path to output result JSON (default: psi4_v2rdm_result.json)",
    )

    args = parser.parse_args()

    # 1. Read input JSON
    if not os.path.exists(args.json):
        print(f"Error: JSON file '{args.json}' not found.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading Hamiltonian and system parameters from: {args.json}")
    with open(args.json, "r") as f:
        data = json.load(f)

    N = int(data["N"])
    M = int(data["M"])
    g = float(data.get("g", 1.0))
    T = np.array(data["T"], dtype=float)
    V = np.array(data["V"], dtype=float)
    K = np.array(data["K"], dtype=float) if "K" in data else None

    exact_energy = None
    if "eigenenergies" in data and len(data["eigenenergies"]) > 0:
        exact_energy = float(min(data["eigenenergies"]))

    print(f"System: N = {N} particles, M = {M} single-particle states, g = {g}")
    print(f"Matrix dimensions: T is ({T.shape[0]}x{T.shape[1]}), V is ({V.shape[0]}x{V.shape[1]})")
    if exact_energy is not None:
        print(f"Exact FCI ground state energy from JSON: {exact_energy:.8f}")

    # 2. Write FCIDUMP file
    fcidump_filename = args.fcidump if args.fcidump else f"{os.path.splitext(args.json)[0]}.fcidump"
    write_fcidump(fcidump_filename, N, M, T, V, g)

    # 3. Determine execution mode and run solver
    result = None

    if args.mode == "psi4":
        result = run_psi4_v2rdm(N, M, g, T, V, positivity=args.positivity, fcidump_path=fcidump_filename)
    elif args.mode == "direct":
        result = run_direct_sdp_cvxopt(N, M, g, T, V, K=K, positivity=args.positivity)
    else: # auto
        try:
            import psi4
            result = run_psi4_v2rdm(N, M, g, T, V, positivity=args.positivity, fcidump_path=fcidump_filename)
        except ImportError as e:
            print("\n[Notice] Psi4 / v2rdm_casscf not detected in this Python environment.")
            print("         Running via built-in high-precision SDP solver (CVXOPT)...")
            result = run_direct_sdp_cvxopt(N, M, g, T, V, K=K, positivity=args.positivity)

    # 4. Compare with exact energy and save output
    if exact_energy is not None:
        err = abs(result['energy'] - exact_energy)
        print(f"\nExact FCI Energy:  {exact_energy:14.8f}")
        print(f"Calculated SDP:    {result['energy']:14.8f}")
        print(f"Absolute Diff:     {err:14.8f}")

    result["N"] = N
    result["M"] = M
    result["g"] = g
    if exact_energy is not None:
        result["exact_energy"] = exact_energy

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nResults saved to: {args.out}")


if __name__ == "__main__":
    main()
