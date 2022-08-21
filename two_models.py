import numpy as np
from percentiles2d import plot_cov_ellipse


def model(x, a, b):
    # two-parameter model.
    return a * x + b


def generate_data(xs, pars, sigma):
    #  Make synthetic data from model and noise
    return model(xs, *pars) + np.random.normal(0.0, sigma, xs.size)


def plot(xpoints, ypoints, ax, marker="x", color="blue", linewidth=0, label=""):
    ax.plot(
        xpoints,
        ypoints,
        marker=marker,
        linewidth=linewidth,
        color=color,
        label=label,
        markerfacecolor="none",
    )


def plot_parameters(popt, pcov, ax, name, idx_x=0, idx_y=1):
    ax.plot(
        popt[idx_x],
        popt[idx_y],
        "o",
        label=f"MLE {name}: {np.array2string(popt, precision=2)}",
        zorder=999,
        color="green",
    )

    nstds = [1, 2]

    for nstd in nstds:
        ellip = plot_cov_ellipse(
            pcov,
            popt,
            ax=ax,
            nstd=nstd,
            label=f"MLE COV @ {nstd}x std",
            fill=None,
            edgecolor="green",
            linewidth=1,
            zorder=999,
        )
        ellip.set(alpha=0.5)
