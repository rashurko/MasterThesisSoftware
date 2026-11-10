import numpy as np
import matplotlib.pyplot as plt
import json

def plotEnergies(input_file="exactDiagSDPcomparison.json", output_file="exactSDPenergies.png"):
    with open(input_file, 'r') as f:
        data = json.load(f)

    N = data['N']
    g = data['g']
    results = data['results']

    M_vals = []
    E_exact_vals = []
    E_SDP_vals = []
    E_proj_vals = []
    E_ADMM_vals = []

    fig, ax = plt.subplots(figsize=(10, 6))

    for result in results:
        M = result['M']
        E_exact = result['E_exact']
        E_SDP = result['E_SDP']

        M_vals.append(M)
        E_exact_vals.append(E_exact)
        E_SDP_vals.append(E_SDP)
        E_proj_vals.append(result.get('E_proj', np.nan))  # Use NaN if E_proj is not available
        E_ADMM_vals.append(result.get('E_ADMM', np.nan))  # Use NaN if E_ADMM is not available

    ax.plot(M_vals, E_exact_vals, linestyle='-', color='black', marker='o', label='ED')
    ax.plot(M_vals, E_SDP_vals, linestyle='--', color='black', marker='x', label='SDP')
    ax.plot(M_vals, E_proj_vals, linestyle='-.', color='black', marker='s', label='Projection')
    ax.plot(M_vals, E_ADMM_vals, linestyle=':', color='black', marker='^', label='ADMM')

    ax.set_xlabel('M', fontsize=14)
    ax.set_ylabel('Energy', fontsize=14)
    ax.set_title(f'Exact Diagonalization vs SDP vs Projection vs ADMM for N={N}, g={g}', fontsize=16)
    ax.legend(fontsize=12)

    plt.savefig(output_file)
    plt.close()

def plotCompTime(input_file="exactDiagSDPcomparison.json", output_file="exactSDPcompTime.png"):
    with open(input_file, 'r') as f:
        data = json.load(f)

    N = data['N']
    g = data['g']
    results = data['results']

    M_vals = []
    time_exact_vals = []
    time_SDP_vals = []
    time_proj_vals = []
    time_ADMM_vals = []

    fig, ax = plt.subplots(figsize=(10, 6))

    for result in results:
        M = result['M']
        time_exact = result['durationDiag']
        time_SDP = result['durationPr']

        M_vals.append(M)
        time_exact_vals.append(time_exact)
        time_SDP_vals.append(time_SDP)
        time_proj_vals.append(result.get('durationProj', np.nan))  # Use NaN if durationProj is not available
        time_ADMM_vals.append(result.get('durationADMM', np.nan))  # Use NaN if durationADMM is not available

    ax.plot(M_vals, time_exact_vals, linestyle='-', color='black', marker='o', label='ED')
    ax.plot(M_vals, time_SDP_vals, linestyle='--', color='black', marker='x', label='SDP')
    ax.plot(M_vals, time_proj_vals, linestyle='-.', color='black', marker='s', label='Projection')
    ax.plot(M_vals, time_ADMM_vals, linestyle=':', color='black', marker='^', label='ADMM')

    #Plot theoretical scaling of O(M^6)
    M_theory = np.array(M_vals)
    time_theory = 1e-6 * M_theory**6  # Adjust
    ax.plot(M_vals, time_theory, linestyle=':', color='black', label='Theoretical O(M^6)')
    # Theoretical scaling of O(M^4)
    time_theory_M4 = 1e-6 * M_theory**4  # Adjust
    ax.plot(M_vals, time_theory_M4, linestyle=':', color='gray', label='Theoretical O(M^4)')

    # Set logarithmic scale for y-axis
    ax.set_yscale('log')

    ax.set_xlabel('M', fontsize=14)
    ax.set_ylabel('Duration (s)', fontsize=14)
    ax.set_title(f'Duration: Exact Diagonalization vs SDP vs Projection vs ADMM for N={N}, g={g}', fontsize=16)
    ax.legend(fontsize=12)

    plt.savefig(output_file)
    plt.close()

def plotEnergiesG(input_file="exactDiagSDPcomparison_g.json", output_file="exactSDPenergies_g.png"):
    with open(input_file, 'r') as f:
        data = json.load(f)

    N = data['N']
    M = data['M']
    results = data['results']

    g_vals = []
    E_exact_vals = []
    E_SDP_vals = []
    E_proj_vals = []
    E_ADMM_vals = []

    fig, ax = plt.subplots(figsize=(10, 6))

    for result in results:
        g = result['g']
        E_exact = result['E_exact']
        E_SDP = result['E_SDP']

        g_vals.append(g)
        E_exact_vals.append(E_exact)
        E_SDP_vals.append(E_SDP)
        E_proj_vals.append(result.get('E_proj', np.nan))  # Use NaN if E_proj is not available
        E_ADMM_vals.append(result.get('E_ADMM', np.nan))  # Use NaN if E_ADMM is not available
    ax.plot(g_vals, E_exact_vals, linestyle='-', color='black', marker='o', label='ED')
    ax.plot(g_vals, E_SDP_vals, linestyle='--', color='black', marker='x', label='SDP')
    ax.plot(g_vals, E_proj_vals, linestyle='-.', color='black', marker='s', label='Projection')
    ax.plot(g_vals, E_ADMM_vals, linestyle=':', color='black', marker='^', label='ADMM')

    ax.set_xlabel('g', fontsize=14)
    ax.set_ylabel('Energy', fontsize=14)
    ax.set_title(f'Exact Diagonalization vs SDP vs Projection vs ADMM for N={N}, M={M}', fontsize=16)
    ax.legend(fontsize=12)

    plt.savefig(output_file)
    plt.close()

# Plots the norms Gamma_exact - Gamma_SDP and Gamma_exact - Gamma_proj as a function of M for fixed N and g
def plotDiffNorm(input_file="exactDiagSDPcomparison.json", output_file="exactSDPdiffNorm.png"):
    with open(input_file, 'r') as f:
        data = json.load(f)

    N = data['N']
    g = data['g']
    results = data['results']

    M_vals = []
    norm_exact_SDP_vals = []
    norm_exact_proj_vals = []
    norm_exact_ADMM_vals = []

    fig, ax = plt.subplots(figsize=(10, 6))

    for result in results:
        M = result['M']

        # Calculate the norms out of Gamma matrices
        Gamma_exact = np.array(result['Gamma_exact'])
        Gamma_SDP = np.array(result['Gamma_SDP'])
        norm_exact_SDP = np.linalg.norm(Gamma_exact - Gamma_SDP)

        Gamma_proj = np.array(result.get('Gamma_proj', np.zeros_like(Gamma_exact)))  # Use zeros if Gamma_proj is not available
        norm_exact_proj = np.linalg.norm(Gamma_exact - Gamma_proj)

        Gamma_ADMM = np.array(result.get('Gamma_ADMM', np.zeros_like(Gamma_exact)))  # Use zeros if Gamma_ADMM is not available
        norm_exact_ADMM = np.linalg.norm(Gamma_exact - Gamma_ADMM)

        M_vals.append(M)
        norm_exact_SDP_vals.append(norm_exact_SDP)
        norm_exact_proj_vals.append(norm_exact_proj)
        norm_exact_ADMM_vals.append(norm_exact_ADMM)

    ax.plot(M_vals, norm_exact_SDP_vals, linestyle='--', color='black', marker='x', label=r'||$\Gamma_{exact} - \Gamma_{SDP}$||')
    ax.plot(M_vals, norm_exact_proj_vals, linestyle='-.', color='black', marker='s', label=r'||$\Gamma_{exact} - \Gamma_{proj}$||')
    ax.plot(M_vals, norm_exact_ADMM_vals, linestyle=':', color='black', marker='^', label=r'||$\Gamma_{exact} - \Gamma_{ADMM}$||')

    ax.set_xlabel('M', fontsize=14)
    ax.set_ylabel('Norm Difference', fontsize=14)
    ax.set_title(f'Norm Differences: Exact Diagonalization vs SDP vs Projection vs ADMM for N={N}, g={g}', fontsize=16)
    ax.legend(fontsize=12)

    plt.savefig(output_file)
    plt.close()


plotEnergies()
plotCompTime()
plotEnergiesG()
plotDiffNorm()
