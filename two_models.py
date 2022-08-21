import numpy as np


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
