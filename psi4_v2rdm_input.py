#!/usr/bin/env python3
r"""
psi4_v2rdm_input.py: Standalone Psi4 driver script for v2rdm-casscf calculations.

Loads custom Hamiltonian integrals (T and V) from exactSol.json and runs
variational 2-RDM driven CASSCF using the v2rdm_casscf plugin in Psi4.

Requirements:
    Psi4 with the v2rdm_casscf plugin installed:
    conda create -n psi4_env -c psi4 -c conda-forge psi4 v2rdm_casscf python=3.10
    conda activate psi4_env
"""

import json
import os
import sys
import numpy as np

try:
    import psi4
except ImportError:
    print("Error: Psi4 is not installed in the active Python environment.", file=sys.stderr)
    print("Run this script inside your Psi4 conda environment, or use ./run_psi4_v2rdm.py for direct SDP.", file=sys.stderr)
    sys.exit(1)


def main():
    json_path = "exactSol.json"
    if len(sys.argv) > 1:
        json_path = sys.argv[1]

    if not os.path.exists(json_path):
        print(f"Error: {json_path} does not exist.", file=sys.stderr)
        sys.exit(1)

    with open(json_path, "r") as f:
        data = json.load(f)

    N = int(data["N"])
    M = int(data["M"])
    g = float(data.get("g", 1.0))
    T = np.array(data["T"], dtype=float)
    V = np.array(data["V"], dtype=float)

    print(f"[Psi4 Input] Loaded system: N={N}, M={M}, g={g}")

    # 1. Output setup
    psi4.core.set_output_file("psi4_v2rdm.out", False)

    # 2. Setup non-interacting dummy geometry for M active orbitals
    geom_str = "\n".join([f"H 0.0 0.0 {i * 100.0}" for i in range(M)])
    mol = psi4.geometry(geom_str)
    mol.set_molecular_charge(0)
    mol.set_multiplicity(1 if N % 2 == 0 else 2)
    mol.update_geometry()

    # 3. Basic SCF setup
    psi4.set_options({
        'basis': 'sto-3g',
        'reference': 'rohf' if N % 2 != 0 else 'rhf',
        'scf_type': 'pk',
        'd_convergence': 1e-8,
        'e_convergence': 1e-8,
    })

    # 4. v2rdm_casscf plugin configuration
    # Enforces DQG conditions (Gamma >= 0, Q >= 0, G >= 0)
    # optimize_orbitals=False ensures the model Hamiltonian basis is preserved
    psi4.set_options({
        'positivity': 'dqg',
        'optimize_orbitals': False,
        'semicanonicalize_orbitals': False,
        'r_convergence': 1e-6,
        'e_convergence': 1e-7,
        'maxiter': 20000,
        'constrain_spin': False,
    })

    # 5. Build Wavefunction and set custom 1-electron Hamiltonian (T)
    wfn = psi4.core.Wavefunction.build(mol, psi4.core.get_global_option('BASIS'))
    H_core = psi4.core.Matrix.from_array(T)
    wfn.set_hcore(H_core)

    # 6. Run variational 2-RDM driven CASSCF
    print("[Psi4] Starting v2rdm-casscf calculation...")
    energy, final_wfn = psi4.energy('v2rdm-casscf', ref_wfn=wfn, return_wfn=True)

    print(f"\n[Psi4] Completed v2rdm-casscf!")
    print(f"Ground state energy (v2rdm-casscf DQG): {energy:.8f}")

    if "eigenenergies" in data and len(data["eigenenergies"]) > 0:
        exact_e = min(data["eigenenergies"])
        print(f"Exact FCI reference energy:             {exact_e:.8f}")
        print(f"Difference:                             {abs(energy - exact_e):.8f}")


if __name__ == "__main__":
    main()

