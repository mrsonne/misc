import numpy as np
from percentiles2d import plot_cov_ellipse


def make_data():
    sigma = 0.25
    pars_model1 = 1.0, 1.0
    # pars_model2 = 0.0, 1.0
    pars_model2 = 0.0, 0.0  # tricky

    print(f"sigma {sigma}")

    # First data set
    xmin, xmax = -1, 1
    sigma_model1 = sigma
    n_model1 = 10
    xs_model1 = np.linspace(xmin, xmax, n_model1)
    ys_model1 = generate_data(xs_model1, pars_model1, sigma_model1)

    # Second data set
    sigma_model2 = sigma
    n_model2 = 10
    xs_model2 = np.linspace(xmin, xmax, n_model2)
    ys_model2 = generate_data(xs_model2, pars_model2, sigma_model2)

    xs_all = np.concatenate((xs_model1, xs_model2))
    ys_all = np.concatenate((ys_model1, ys_model2))
    return xs_model1, ys_model1, xs_model2, ys_model2, xs_all, ys_all


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
