import numpy as np


def model(x, a, b):
    # two-parameter model.
    return a * x + b


def generate_data(xs, pars, sigma):
    #  Make synthetic data from model and noise
    return model(xs, *pars) + np.random.normal(0.0, sigma, xs.size)


def plot(xpoints, ypoints, ax, marker="x", color="blue"):
    ax.plot(xpoints, ypoints, marker=marker, linewidth=0, color=color)
