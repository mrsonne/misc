import numpy as np
from percentiles2d import plot_cov_ellipse
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import itertools

CMAP_NAME = "tab10"

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


def plot_parameters(
    popt, pcov, ax, name, idx_x=0, idx_y=1, nstds=[1, 2], color="green"
):
    ax.plot(
        popt[idx_x],
        popt[idx_y],
        "o",
        label=f"MLE {name}: {np.array2string(popt, precision=2)}",
        zorder=999,
        color=color,
    )

    for nstd in nstds:
        ellip = plot_cov_ellipse(
            pcov,
            popt,
            ax=ax,
            nstd=nstd,
            label=f"MLE COV @ {nstd}x std",
            facecolor=color,
            alpha=0.25,
            edgecolor=color,
            linewidth=3,
            zorder=999,
        )
        # ellip.set(alpha=0.5)

tol = 1e-15
def pie_scatter(xs, ys, ratios, sizes, size_center, colors, ax):
    assert sum(ratios) <= 1 + tol, f'sum of ratios needs to be < 1 but found {sum(ratios)} = {"+".join([str(r) for r in ratios])}'

    markers = []
    # previous = 0
    previous = np.pi / 2 # start at noon 
    npoints = 500
    # calculate the points of the pie pieces
    for color, ratio in zip(colors, ratios):
        this =  previous - 2 * np.pi * ratio # counter clockwise
        points = np.linspace(previous, this, npoints)
        x  = [0] + np.cos(points).tolist()
        y  = [0] + np.sin(points).tolist()
        xy = np.column_stack([x, y])
        previous = this
        # markers.append({'marker': xy, 's': np.abs(xy).max()**2 * np.array(sizes), 'facecolor': color})
        markers.append({'marker': xy, 's': np.array(sizes), 'facecolor': color})

    # scatter each of the pie pieces to create pies
    for marker in markers:
        ax.scatter(xs, ys, **marker)
    ax.scatter(xs, ys, s=size_center, color="w")


def plot_ncomp(xs, ys, b0, b1, model_fun, cat=None, p_cat=None, p_cat_all=None):
    fig, ax = plt.subplots(figsize=(16, 8), dpi=180)
    size_true = 144
    size_pred = size_true*0.5
    size_pie = size_true*3.5
    xmin = 1e12
    xmax = -xmin

    # For transforming cluster probability to alpha
    if cat is not None:
        amin, amax = 0.25, 1
        pmin, pmax = min(np.min(p_cat), 0.5), 1
        slope = (amax - amin) / (pmax - pmin)

    colors = itertools.cycle(get_cmap(CMAP_NAME).colors)
    # Plot data and true categories
    for icmp, (x, y) in enumerate(zip(xs, ys), 1):
        ax.scatter(x, y, s=size_true, facecolors="none", edgecolors=next(colors), label=f"True CMP {icmp}", zorder=10000)
        xmin = min(xmin, min(x))
        xmax = max(xmax, max(x))

    if p_cat_all is not None:
        # pie chart for each point visualizing the category prevalence
        # https://stackoverflow.com/questions/54541081/how-to-plot-a-donut-chart-around-a-point-on-a-scatterplot
        x_all = np.concatenate(xs)
        y_all = np.concatenate(ys)
        n_cat = p_cat_all.shape[0]
        pie_colors = get_cmap(CMAP_NAME).colors[:n_cat]
        for x, y, frc in zip(x_all, y_all, p_cat_all.transpose()):
            pie_scatter([x], [y], frc, [size_pie], size_true*1.5, pie_colors, ax)


    if cat is not None:
        categories = np.unique(cat)
        x_all = np.concatenate(xs)
        y_all = np.concatenate(ys)
        colors = itertools.cycle(get_cmap(CMAP_NAME).colors)
        for cat_id in categories:

            # Set face alpha to the class probability
            p = p_cat[cat == cat_id]
            alpha = slope * (p - pmin) + amin
            alpha = np.where(alpha < 0, 0, alpha)

            color = next(colors)
            ax.scatter(
                x_all[cat == cat_id],
                y_all[cat == cat_id],
                s=size_pred,
                facecolors=color,
                alpha=alpha,
                edgecolors="none",
                # label=f"Predicted CMP {cat_id}"
            )

            # Fake data point used to set the alpha to 1 in the legend 
            ax.scatter(
                [np.nan],
                [np.nan],
                s=size_pred,
                facecolors=color,
                alpha=1.0,
                edgecolors="none",
                label=f"Predicted CMP {cat_id + 1}"
            )

    colors = itertools.cycle(get_cmap(CMAP_NAME).colors)
    x_pred = np.linspace(xmin, xmax, 10)
    cat_count = np.unique(cat, return_counts=True)[1]
    for icmp, (b0_, b1_) in enumerate(zip(b0, b1), 1):
        y_pred = model_fun(x_pred, b0_, b1_)
        ax.plot(x_pred, y_pred, "-", color=next(colors), label=f"Model {icmp} [{b0_:5.2}, {b1_:5.2}] ({cat_count[icmp-  1]} points)")

    leg = ax.legend()

    # alpha is reset when I add more data to the fig?!?
    # for lh in leg.legendHandles: 
    #     lh.set_alpha(1)

    return fig
