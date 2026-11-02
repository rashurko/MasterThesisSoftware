import os
import glob
import re
import json
import numpy as np
import matplotlib.pyplot as plt


def parse_step_sizes_data(folder="StepSizes"):
    """
    Parses all 'projectionDykstra_stepSize_*.txt' files in the given folder.

    Returns:
        data: dict mapping step_size (float) -> dict with keys:
              'iterations', 'energies', 'norm_diffs', 'norm_deltas'
        meta: dict containing metadata (N, M, g, alpha, E_sdp) if available
    """
    files = sorted(glob.glob(os.path.join(folder, "projectionDykstra_stepSize_*.txt")))

    data = {}
    for f in files:
        m = re.search(r"stepSize_([0-9.]+)\.txt", f)
        if not m:
            continue
        step = float(m.group(1))

        iters, energies, norm_diffs, norm_deltas = [], [], [], []
        with open(f, "r") as fp:
            for line in fp:
                m_iter = re.search(r"Iteration\s+(\d+)", line)
                m_energy = re.search(r"Energy:\s+([-\d.eE+]+)", line)
                m_norm = re.search(r"Norm diff:\s+([-\d.eE+]+)", line)
                m_delta = re.search(r"Norm delta Gamma:\s+([-\d.eE+]+)", line)

                if m_iter and m_energy and m_norm:
                    iters.append(int(m_iter.group(1)))
                    energies.append(float(m_energy.group(1)))
                    norm_diffs.append(float(m_norm.group(1)))
                    norm_deltas.append(
                        float(m_delta.group(1)) if m_delta else np.nan
                    )

        if iters:
            data[step] = {
                "iterations": np.array(iters),
                "energies": np.array(energies),
                "norm_diffs": np.array(norm_diffs),
                "norm_deltas": np.array(norm_deltas),
            }

    # Extract metadata from JSON if present
    meta = {}
    json_path = os.path.join(folder, "projectionMethodMultipleStepSizes.json")
    if os.path.exists(json_path):
        try:
            with open(json_path, "r") as fp:
                json_data = json.load(fp)
                meta["N"] = json_data.get("N")
                meta["M"] = json_data.get("M")
                meta["g"] = json_data.get("g")
                meta["alpha"] = json_data.get("alpha")
                for r in json_data.get("results", []):
                    if r.get("step_size") == "SDP":
                        meta["E_sdp"] = r.get("E_proj")
        except Exception:
            pass

    return data, meta


def plotStepSizes(
    folder="StepSizes",
    output_file="projectionDykstra_stepSizes.png",
    max_iter=None,
    x_log=False,
    y_log_norm=True,
    y_log_delta=True,
    skip_first_norm=True,
    show_sdp=True,
):
    r"""
    Plots:
      1. Energy vs. Iteration (top)
      2. Norm difference ||D_n - D_{n-1}|| vs. Iteration (middle)
      3. Norm delta Gamma ||\Delta \Gamma|| vs. Iteration (bottom, under norm diff)

    Parameters:
        folder (str): Directory containing step size txt files.
        output_file (str): Filepath to save the generated plot.
        max_iter (int, optional): If set, limits x-axis to [0, max_iter].
        x_log (bool): If True, uses a logarithmic scale for the iteration axis.
        y_log_norm (bool): If True, uses logarithmic scale for norm diff.
        y_log_delta (bool): If True, uses logarithmic scale for norm delta Gamma.
        skip_first_norm (bool): If True, skips iteration 0 for norm diff to prevent
                                initial condition scale distortion.
        show_sdp (bool): If True, displays the SDP ground state energy reference line.
    """
    data, meta = parse_step_sizes_data(folder)

    if not data:
        print(f"No step size data files found in '{folder}'.")
        return

    steps = sorted(data.keys())

    # Check whether norm_deltas is available in the parsed data
    has_delta = any(
        not np.all(np.isnan(data[s]["norm_deltas"])) for s in steps
    )

    num_subplots = 3 if has_delta else 2
    fig_height = 11 if has_delta else 8
    fig, axes = plt.subplots(num_subplots, 1, figsize=(10, fig_height), sharex=True)

    if has_delta:
        ax1, ax2, ax3 = axes
    else:
        ax1, ax2 = axes
        ax3 = None

    # Color gradient across step sizes
    cmap = plt.cm.viridis
    colors = [cmap(i / max(1, len(steps) - 1)) for i in range(len(steps))]

    for step, color in zip(steps, colors):
        iters = data[step]["iterations"]
        energies = data[step]["energies"]
        norm_diffs = data[step]["norm_diffs"]
        norm_deltas = data[step]["norm_deltas"]

        # Omit iteration 0 if log scale on x-axis is requested
        if x_log:
            mask = iters >= 1
            iters = iters[mask]
            energies = energies[mask]
            norm_diffs = norm_diffs[mask]
            norm_deltas = norm_deltas[mask]

        # 1. Top subplot: Energy
        ax1.plot(
            iters,
            energies,
            label=f"Step {step:.1f}",
            color=color,
            linewidth=1.5,
        )

        # 2. Middle subplot: Norm difference
        if skip_first_norm and len(iters) > 1 and iters[0] == 0:
            norm_iters = iters[1:]
            norm_vals = norm_diffs[1:]
        else:
            norm_iters = iters
            norm_vals = norm_diffs

        ax2.plot(
            norm_iters,
            norm_vals,
            label=f"Step {step:.1f}",
            color=color,
            linewidth=1.5,
        )

        # 3. Bottom subplot: Norm delta Gamma (under norm diff)
        if ax3 is not None and not np.all(np.isnan(norm_deltas)):
            ax3.plot(
                iters,
                norm_deltas,
                label=f"Step {step:.1f}",
                color=color,
                linewidth=1.5,
            )

    # Reference line for SDP energy
    if show_sdp and "E_sdp" in meta and meta["E_sdp"] is not None:
        e_sdp = meta["E_sdp"]
        ax1.axhline(
            e_sdp,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=f"SDP ({e_sdp:.4f})",
        )

    # Reference line for convergence tolerance in norm diff
    ax2.axhline(
        1e-6,
        color="black",
        linestyle=":",
        linewidth=1.2,
        alpha=0.7,
        label=r"Tolerance ($10^{-6}$)",
    )

    # Set scales
    bottom_ax = ax3 if ax3 is not None else ax2

    if x_log:
        ax1.set_xscale("log")
        ax2.set_xscale("log")
        if ax3 is not None:
            ax3.set_xscale("log")
        bottom_ax.set_xlabel("Iteration Number (log scale)", fontsize=13)
    else:
        bottom_ax.set_xlabel("Iteration Number", fontsize=13)

    if y_log_norm:
        ax2.set_yscale("log")

    if ax3 is not None and y_log_delta:
        ax3.set_yscale("log")

    if max_iter is not None:
        bottom_ax.set_xlim(0, max_iter)

    # Labels and Titles
    ax1.set_ylabel("Energy", fontsize=13)
    ax2.set_ylabel(r"Norm Difference $||D_n - D_{n-1}||$", fontsize=13)
    if ax3 is not None:
        ax3.set_ylabel(r"Norm $\Delta \Gamma$ ($||\Delta \Gamma||$)", fontsize=13)

    title_parts = ["Dykstra's Projection Method for Different Step Sizes"]
    meta_info = []
    if "N" in meta and meta["N"] is not None:
        meta_info.append(f"N={meta['N']}")
    if "M" in meta and meta["M"] is not None:
        meta_info.append(f"M={meta['M']}")
    if "g" in meta and meta["g"] is not None:
        meta_info.append(f"g={meta['g']}")
    if "alpha" in meta and meta["alpha"] is not None:
        meta_info.append(rf"$\alpha$={meta['alpha']}")

    if meta_info:
        title_parts.append(f"({', '.join(meta_info)})")

    ax1.set_title(" ".join(title_parts), fontsize=15, pad=10)

    # Grids and legends
    ax1.grid(True, which="both", linestyle="--", alpha=0.5)
    ax2.grid(True, which="both", linestyle="--", alpha=0.5)
    if ax3 is not None:
        ax3.grid(True, which="both", linestyle="--", alpha=0.5)

    ax1.legend(fontsize=9, ncol=2, loc="best")
    ax2.legend(fontsize=9, ncol=2, loc="best")
    if ax3 is not None:
        ax3.legend(fontsize=9, ncol=2, loc="best")

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plot saved successfully to: {output_file}")


if __name__ == "__main__":
    # 1. Full run across all iterations
    plotStepSizes(
        folder="StepSizes",
        output_file="StepSizes/projectionDykstra_stepSizes.png",
    )

    # 2. Zoomed-in plot focusing on early convergence (e.g. first 600 iterations)
    plotStepSizes(
        folder="StepSizes",
        output_file="StepSizes/projectionDykstra_stepSizes_zoom.png",
        max_iter=600,
    )

    # 3. Logarithmic x-axis view to display both early dynamics and late iterations
    plotStepSizes(
        folder="StepSizes",
        output_file="StepSizes/projectionDykstra_stepSizes_logx.png",
        x_log=True,
    )
