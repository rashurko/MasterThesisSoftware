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

    fig, ax = plt.subplots(figsize=(10, 6))

    for result in results:
        M = result['M']
        E_exact = result['E_exact']
        E_SDP = result['E_SDP']

        M_vals.append(M)
        E_exact_vals.append(E_exact)
        E_SDP_vals.append(E_SDP)

    ax.plot(M_vals, E_exact_vals, linestyle='-', color='black', marker='o', label='ED')
    ax.plot(M_vals, E_SDP_vals, linestyle='--', color='black', marker='x', label='SDP')

    ax.set_xlabel('M', fontsize=14)
    ax.set_ylabel('Energy', fontsize=14)
    ax.set_title(f'Exact Diagonalization vs SDP for N={N}, g={g}', fontsize=16)
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

    fig, ax = plt.subplots(figsize=(10, 6))

    for result in results:
        M = result['M']
        time_exact = result['durationDiag']
        time_SDP = result['durationPr']

        M_vals.append(M)
        time_exact_vals.append(time_exact)
        time_SDP_vals.append(time_SDP)

    ax.plot(M_vals, time_exact_vals, linestyle='-', color='black', marker='o', label='ED')
    ax.plot(M_vals, time_SDP_vals, linestyle='--', color='black', marker='x', label='SDP')

    # Set logarithmic scale for y-axis
    ax.set_yscale('log')

    ax.set_xlabel('M', fontsize=14)
    ax.set_ylabel('Duration (s)', fontsize=14)
    ax.set_title(f'Duration: Exact Diagonalization vs SDP for N={N}, g={g}', fontsize=16)
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

    fig, ax = plt.subplots(figsize=(10, 6))

    for result in results:
        g = result['g']
        E_exact = result['E_exact']
        E_SDP = result['E_SDP']

        g_vals.append(g)
        E_exact_vals.append(E_exact)
        E_SDP_vals.append(E_SDP)

    ax.plot(g_vals, E_exact_vals, linestyle='-', color='black', marker='o', label='ED')
    ax.plot(g_vals, E_SDP_vals, linestyle='--', color='black', marker='x', label='SDP')

    ax.set_xlabel('g', fontsize=14)
    ax.set_ylabel('Energy', fontsize=14)
    ax.set_title(f'Exact Diagonalization vs SDP for N={N}, M={M}', fontsize=16)
    ax.legend(fontsize=12)

    plt.savefig(output_file)
    plt.close()


plotEnergies()
plotCompTime()
plotEnergiesG()
