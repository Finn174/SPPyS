"""
plotting.py

Functions for plotting simulation and fitting results.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.ticker import (
    MaxNLocator,
    AutoMinorLocator,
    ScalarFormatter,
)
from matplotlib.colors import from_levels_and_colors

from model import (
    generate_grid,
    simulate_image,
)


# --------------------------------------------------------
# Tick formatting
# --------------------------------------------------------

def apply_clean_ticks(
    ax,
    xlim,
    ylim,
    n_xticks=5,
    n_yticks=5,
    sci=False,
):
    """
    Applies clean tick formatting to improve readability.
    """

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    ax.xaxis.set_major_locator(
        MaxNLocator(
            nbins=n_xticks,
            min_n_ticks=2,
            steps=[1, 2, 2.5, 5, 10],
        )
    )

    ax.yaxis.set_major_locator(
        MaxNLocator(
            nbins=n_yticks,
            min_n_ticks=2,
            steps=[1, 2, 2.5, 5, 10],
        )
    )

    ax.xaxis.set_minor_locator(
        AutoMinorLocator(2)
    )

    ax.yaxis.set_minor_locator(
        AutoMinorLocator(2)
    )

    formatter = ScalarFormatter(
        useMathText=True,
        useOffset=False,
    )

    if sci:
        formatter.set_powerlimits((-3, 3))
    else:
        formatter.set_powerlimits((0, 0))

    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)


# --------------------------------------------------------
# Getting axis labels from Q-vectors
# --------------------------------------------------------

def axis_labels_from_qvectors(Qin,Qout):
    """
    Generates axis labels from Qin and Qout.
    """

    x_label = (
        f"[{int(Qin[0])} {int(Qin[1])} {int(Qin[2])}]"
    )

    y_label = (
        f"[{int(Qout[0])} {int(Qout[1])} {int(Qout[2])}]"
    )

    return x_label, y_label


# --------------------------------------------------------
# Plotting a simulation
# --------------------------------------------------------

def plot_simulation(
    params,
    npoints=100,
):
    """
    Plots a simulated reciprocal-space map.

    Parameters
    ----------
    params : SimulationParameters

    npoints : int
        Number of grid points.
    """

    xs, ys = generate_grid(
        params,
        npoints=npoints,
    )

    image = simulate_image(
        xs,
        ys,
        params,
    )

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    if params.Sub_peak is None:

        plot_max = (
            params.background
            + 2 * params.film_intensity
        )

    else:
        if params.sub_intensity < params.film_intensity:
            plot_max = 2 * params.film_intensity + params.background
        else:
            plot_max = (2 * params.film_intensity + params.background + params.sub_intensity) / params.axis_compress

    contour = ax.contourf(
        xs,
        ys,
        image,
        cmap="inferno",
        levels=9,
        vmin=params.background,
        vmax=plot_max,
    )

    cbar = fig.colorbar(
        contour,
        pad=0.08,
    )

    cbar.set_label(
        "Intensity (counts)",
        fontsize=18,
    )

    cbar.locator = ticker.MaxNLocator(
        nbins=5
    )

    cbar.ax.tick_params(labelsize=18)

    cbar.update_ticks()

    if params.xlabel is None:
        params.xlabel = axis_labels_from_qvectors(params.Qin, params.Qout)[0]

    if params.ylabel is None:
        params.ylabel = axis_labels_from_qvectors(params.Qin, params.Qout)[1]

    ax.set_xlabel(
        params.xlabel,
        fontsize=18,
    )

    ax.set_ylabel(
        params.ylabel,
        fontsize=18,
    )

    xlim = (
        xs.min(),
        xs.max(),
    )

    ylim = (
        ys.min(),
        ys.max(),
    )

    apply_clean_ticks(
        ax,
        xlim,
        ylim,
        n_xticks=4,
        n_yticks=4,
        sci=True,
    )

    ax.tick_params(
        labelsize=18
    )

    plt.tight_layout()

    return fig, ax


def plot_fit_results(
    params,
    xs,
    ys,
    experimental,
    fitted,
    output_filename,
    red_chi2,
):
    """
    Plots an experimental data file, the fitted simulation,
    and residuals side-by-side.
    """

    residual = fitted - experimental

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 5),
        constrained_layout=True,
    )

    xlim = (
        xs.min(),
        xs.max(),
    )
    
    ylim = (
        ys.min(),
        ys.max(),
    )

    num_levels = 40
    res_vmin = residual.min()
    res_vmax = residual.max()
    midpoint = 0
    res_levels = np.linspace(res_vmin, res_vmax, num_levels)
    res_midp = np.mean(np.c_[res_levels[:-1], res_levels[1:]], axis=1)
    res_vals = np.interp(res_midp, [res_vmin, midpoint, res_vmax], [0, 0.5, 1])
    res_colors = plt.cm.bwr(res_vals)
    res_cmap, res_norm = from_levels_and_colors(res_levels, res_colors)

    # -------------------------------
    # Experimental data plot
    # -------------------------------

    im0 = axes[0].contourf(
        xs,
        ys,
        experimental,
        levels=num_levels,
        cmap="inferno",
    )

    axes[0].set_title("Experimental", fontsize=18)

    if params.xlabel is None:
        params.xlabel = axis_labels_from_qvectors(params.Qin, params.Qout)[0]
    
    if params.ylabel is None:
        params.ylabel = axis_labels_from_qvectors(params.Qin, params.Qout)[1]

    axes[0].set_xlabel(
        params.xlabel,
        fontsize=18,
    )

    axes[0].set_ylabel(
        params.ylabel,
        fontsize=18,
    )

    apply_clean_ticks(
        axes[0],
        xlim,
        ylim,
        n_xticks=4,
        n_yticks=4,
        sci=True,
    )

    axes[0].tick_params(
        labelsize=18
    )

    # -------------------------------
    # Fitted simulation plot
    # -------------------------------

    im1 = axes[1].contourf(
        xs,
        ys,
        fitted,
        levels=num_levels,
        cmap="inferno",
    )

    axes[1].set_title("Fitted Simulation", fontsize=18)

    axes[1].set_xlabel(
        params.xlabel,
        fontsize=18,
    )

    cbar = fig.colorbar(im0, ax=[axes[0], axes[1]])
    cbar.ax.tick_params(labelsize=18)
    cbar.locator = ticker.MaxNLocator(nbins=5)
    cbar.update_ticks()

    apply_clean_ticks(
        axes[1],
        xlim,
        ylim,
        n_xticks=4,
        n_yticks=4,
        sci=True,
    )
    
    axes[1].tick_params(
        labelsize=18
    )

    axes[1].text(
        xs.min(), ys.max(),
        "Reduced $\chi^2$ = {:.3f}".format(red_chi2),
        fontsize=18, 
        color="red", 
        ha="left",
        va="top",
        bbox=dict(facecolor="white", alpha=0.6, edgecolor="black", pad=0)
    )

    # -------------------------------
    # Residual plot
    # -------------------------------

    im2 = axes[2].contourf(
        xs,
        ys,
        residual,
        levels=num_levels,
        cmap=res_cmap,
    )

    axes[2].set_title("Residual", fontsize=18)

    axes[2].set_xlabel(
        params.xlabel,
        fontsize=18,
    )

    cbar = fig.colorbar(im2, ax=axes[2])
    cbar.ax.tick_params(labelsize=18)
    cbar.ax.set_ylabel('Intensity (counts)',fontsize = 18)
    cbar.locator = ticker.MaxNLocator(nbins=5)
    cbar.update_ticks()

    apply_clean_ticks(
        axes[2],
        xlim,
        ylim,
        n_xticks=4,
        n_yticks=4,
        sci=True,
    )

    axes[2].tick_params(
        labelsize=18
    )

    plt.savefig(f"fit_result_{output_filename}.png", dpi=150, bbox_inches='tight')

    plt.show()