# %% Imports

from typing import Optional
import pickle
import itertools

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pymc3 as pm
import pymc3.distributions.transforms as tr
import seaborn as sns
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from pathlib import Path

# from PIL import ImageColor
import matplotlib.colors as mcolors
from matplotlib.cm import get_cmap
import theano.tensor as tt
import arviz as az
import statsmodels.api as sm

from two_models import plot_parameters


THIS_PATH = Path(__file__).parent
TMP_PATH = THIS_PATH.joinpath("tmp")
TMP_PATH.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = THIS_PATH.joinpath("results")
RESULTS_PATH.mkdir(parents=True, exist_ok=True)
NORMAL_FONTSIZE = 20
RANDOM_SEED = 17
CMAP_NAME = "tab10"
from two_models import plot_cov_ellipse


def model(x, b0, b1):
    # two-parameter model.
    return b1 * x + b0


def plot_bayesian_fit(xs, ys, nsteps=25000, do_ppc: bool = True):
    # Bayesian inference for the true classes
    n_components = 1
    point_estimate = "mean"
    var_names = ["b0", "b1", "sigma"]
    var_names_pairs = ["b0", "b1"]

    fig_posterior, axs = plt.subplots(len(xs), len(var_names), figsize=(20, 20))
    fig_pair, axs_pair = plt.subplots(len(xs), 1, figsize=(20, 20))

    hdi_prob = az.rcParams["stats.hdi_prob"]

    if do_ppc is not None:
        fig_gppc, axs_gppc = plt.subplots(len(xs), 1, figsize=(20, 20))
        fig_mppc, axs_mppc = plt.subplots(len(xs), 1, figsize=(20, 20))
    else:
        fig_gppc = None
        fig_mppc = None

    for i, (x, y) in enumerate(zip(xs, ys)):

        trace, model = fit(
            x,
            y,
            n_components,
            nsteps=nsteps,
            return_inferencedata=True,
            save_trace=False,
        )

        az.plot_posterior(
            trace,
            var_names=var_names,
            point_estimate=point_estimate,
            ax=axs[i, :],
        )

        az.plot_pair(
            trace,
            var_names=var_names_pairs,
            kind="kde",
            point_estimate="mean",
            divergences=True,
            textsize=18,
            ax=axs_pair[i],
        )

        if do_ppc:
            with model:
                ppc = pm.sample_posterior_predictive(
                    trace, var_names=["b0", "b1", "y"], random_seed=RANDOM_SEED
                )

            az.plot_ppc(
                az.from_pymc3(posterior_predictive=ppc, model=model), ax=axs_gppc[i]
            )

            mu_pp = (ppc["b0"] + ppc["b1"] * x[:, None]).T
            axs_mppc[i].plot(x, y, "o", ms=4, alpha=0.4, label="Data")
            axs_mppc[i].plot(x, mu_pp.mean(0), label="Mean outcome", alpha=0.6)
            az.plot_hdi(
                x,
                mu_pp,
                ax=axs_mppc[i],
                fill_kwargs={"alpha": 0.8, "label": f"Mean outcome {hdi_prob} % HPD"},
            )

            az.plot_hdi(
                x,
                ppc["y"],
                ax=axs_mppc[i],
                fill_kwargs={
                    "alpha": 0.8,
                    "color": "#a1dab4",
                    "label": f"Outcome {hdi_prob} % HPD",
                },
            )

            axs_mppc[i].set_xlabel("Predictor")
            axs_mppc[i].set_ylabel("Outcome")
            axs_mppc[i].set_title("Posterior predictive checks")
            axs_mppc[i].legend(ncol=2, fontsize=12)

    return fig_posterior, fig_pair, fig_gppc, fig_mppc


def plot_curve_fit(xs, ys):
    fig, axs = plt.subplots(2, 1, figsize=(20, 20))

    colors = itertools.cycle(get_cmap(CMAP_NAME).colors)
    for i, (x, y) in enumerate(zip(xs, ys), 1):
        popt, pcov = curve_fit(model, x, y)

        # Standard errors
        perr = np.sqrt(np.diag(pcov))

        # sigma
        sigma = np.std(model(x, *popt) - y)

        x_ = sm.add_constant(x)
        lin_model = sm.OLS(y, x_)
        results = lin_model.fit()
        # print(results.summary())
        print("Statsmodels")
        print("params", results.params)
        print("std err", results.bse)
        print("cov1", results.cov_params())

        print("SciPy Opt")
        print("popt", popt)
        print("perr", perr)
        print("pcov", pcov)
        print("sigma", sigma)

        color = next(colors)

        sns.regplot(
            x=x,
            y=y,
            ci=95,
            order=1,
            line_kws={
                "label": f"CMP{i}. MLE={popt} (σ={sigma})",
                "color": color,
            },
            scatter_kws={
                "s": 144,
            },
            seed=1,
            label=f"CMP{i} data.",
            truncate=False,
            color=color,
            ax=axs[0],
        )

        plot_parameters(popt, pcov, axs[1], f"CMP{i}", nstds=[2], color=color)

    axs[0].set_xlabel("x", fontsize=NORMAL_FONTSIZE)
    axs[0].set_ylabel("y", fontsize=NORMAL_FONTSIZE)
    axs[0].legend(fontsize=20)

    axs[1].legend(fontsize=20)
    axs[1].set_xlabel("b0", fontsize=NORMAL_FONTSIZE)
    axs[1].set_ylabel("b1", fontsize=NORMAL_FONTSIZE)

    for ax in axs.flat:
        ax.tick_params(axis="both", which="major", labelsize=NORMAL_FONTSIZE)

    fig.suptitle("MLE parameters inferred from true classes")
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])

    return fig


def c1(X1, Y, return_inferencedata: bool = False):
    """One-component "mixture" """
    nsample = 5000
    nchains = 4
    model = pm.Model()
    with model:
        b0 = pm.Normal("b0", 0, sigma=20)
        b1 = pm.Normal("b1", 0, sigma=20)
        y_est = b0 + b1 * X1
        sigma = pm.HalfCauchy("sigma", beta=10, testval=1.0)
        likelihood = pm.Normal("y", mu=y_est, sigma=sigma, observed=Y)

        trace = pm.sample(
            nsample, return_inferencedata=return_inferencedata, chains=nchains
        )
        return trace, model


def c2(X1, Y, return_inferencedata: bool = False):
    """Two component model that uses BinaryMetropolis
    Modified from https://stats.stackexchange.com/questions/185812/regression-mixture-in-pymc3
    """
    n_components = 2
    size = X1.size
    basic_model = pm.Model()
    with basic_model:
        p = pm.Uniform("p", 0, 1)  # Proportion in each mixture
        b0 = pm.Normal("b0", mu=0, sd=10)  # Intercept
        b1 = pm.Normal("b1", mu=0, sd=100, shape=n_components)  # slopes
        sigma = pm.Uniform("sigma", 0, 20)  # Noise
        # sigma = pm.HalfCauchy("sigma", beta=10, testval=1.0)

        # https://docs.pymc.io/en/v3/pymc-examples/examples/mixture_models/gaussian_mixture_model.html
        # Break the symmetry. b1 array should always be sorted in
        # increasing order so it's easier to compare and average over chains

        if n_components > 1:
            switches = tt.switch(b1[1] - b1[0] < 0, -np.inf, 0)
            order_slopes_potential = pm.Potential(
                "order_slopes_potential",
                switches,
            )

        category = pm.Bernoulli(
            "category", p=p, shape=size
        )  # Classification of each observation

        b1_ = pm.Deterministic("b1_", b1[category])  # Choose b1 based on category

        mu = b0 + b1_ * X1  # Expected value of outcome

        # Likelihood
        likelihood = pm.Normal("y", mu=mu, sd=sigma, observed=Y)

    with basic_model:
        step1 = pm.Metropolis([p, b0, b1, sigma])
        # non-default scaling is important
        step2 = pm.BinaryMetropolis([category], scaling=0.01)

        # the classification part converges to different categories
        trace = pm.sample(
            20000,
            [step1, step2],
            progressbar=True,
            return_inferencedata=return_inferencedata,
            initvals={"b1": np.linspace(-1, 1, n_components)},  #
            # tune=10000,
        )

    return trace


def cn(
    X1,
    Y,
    n_components,
    return_inferencedata: bool = False,
    favor_few_components: bool = True,
    p_min: Optional[float] = 0.1,
    nsteps: int = 10000,
):
    """Modified from
    https://docs.pymc.io/en/v3/pymc-examples/examples/mixture_models/gaussian_mixture_model.html
    """
    assert n_components >= 2, "Must have more than 1 component"
    size = X1.size
    model = pm.Model()
    with model:
        # cluster sizes

        if favor_few_components:
            # prior that should favor populating few dominant components (Gelman et al p 536)
            p = pm.Dirichlet(
                "p", a=np.ones(n_components) / n_components, shape=n_components
            )
        else:
            # fill all components
            p = pm.Dirichlet("p", a=np.ones(n_components), shape=n_components)

            if p_min is not None:
                # ensure all clusters have some points
                p_min_potential = pm.Potential(
                    "p_min_potential", tt.switch(tt.min(p) < p_min, -np.inf, 0)
                )

        b0 = pm.Normal("b0", mu=0, sd=10)  # Intercept
        b1 = pm.Normal(
            "b1",
            mu=0,
            sd=100,
            shape=n_components,
            transform=tr.ordered,
            testval=np.linspace(-1, 1, n_components),
        )
        sigma = pm.Uniform("sigma", 0, 20)  # Noise

        # Deal with identifiability by enforcing a order in slopes
        # switches = tt.switch(b1[1] - b1[0] < 0, -np.inf, 0)
        # for icmp in range(1, n_components - 1):
        #     switches += tt.switch(b1[icmp + 1] - b1[icmp] < 0, -np.inf, 0)

        # order_slopes_potential = pm.Potential(
        #     "order_slopes_potential",
        #     switches,
        # )

        # latent cluster of each observation
        category = pm.Categorical("category", p=p, shape=size)

        b1_ = pm.Deterministic("b1_", b1[category])  # Choose b1 based on category

        mu = b0 + b1_ * X1  # Expected value of outcome

        # Likelihood
        likelihood = pm.Normal("y", mu=mu, sd=sigma, observed=Y)

    # fit model
    with model:
        step1 = pm.Metropolis(vars=[p, b0, b1, sigma])
        step2 = pm.CategoricalGibbsMetropolis(vars=[category])
        # step2 = pm.ElemwiseCategorical(
        #     vars=[category], values=list(range(n_components))
        # )

        trace = pm.sample(
            nsteps,
            step=[step1, step2],
            tune=int(0.2 * (nsteps)),
            progressbar=True,
            return_inferencedata=return_inferencedata,
            # initvals={"b1": np.linspace(-1, 1, n_components)},  #
        )
        return trace, model


def fit(
    X1,
    Y,
    n_components: int,
    return_inferencedata: bool = False,
    favor_few_components: bool = True,
    p_min: Optional[float] = 0.1,
    nsteps: int = 10000,
    save_trace: bool = False,
):
    if n_components == 1:
        trace, model = c1(X1, Y, return_inferencedata=return_inferencedata)
    # elif n_components == 2:
    #     trace = c2(X1, Y)
    elif n_components >= 2:
        trace, model = cn(
            X1,
            Y,
            n_components,
            favor_few_components=favor_few_components,
            return_inferencedata=return_inferencedata,
            p_min=p_min,
            nsteps=nsteps,
        )

    if save_trace:
        with open(TMP_PATH.joinpath(f"az_trace_n{n_components}.pkl"), "wb") as f:
            pickle.dump(trace, f)

    return trace, model


# import seaborn as sns


def make_data():
    np.random.seed(123)
    b0 = 0
    sigma = 2
    b1 = [-5, 5]
    # size = 250
    size1 = 25
    size2 = 35
    size = size1 + size2

    print("size", size1 + size2)
    print("weight1", size1 / size)
    print("weight1", size2 / size)

    # Predictor variable
    # X1_1 = np.random.randn(size)
    X1_1 = np.linspace(-2, 2, size1)

    # Simulate outcome variable--cluster 1
    Y1 = b0 + b1[0] * X1_1 + np.random.normal(loc=0, scale=sigma, size=size1)

    # Predictor variable
    # X1_2 = np.random.randn(size)
    X1_2 = np.linspace(-3, 3, size2)
    # Simulate outcome variable --cluster 2
    Y2 = b0 + b1[1] * X1_2 + np.random.normal(loc=0, scale=sigma, size=size2)

    X1 = np.append(X1_1, X1_2)
    Y = np.append(Y1, Y2)

    return X1, Y, b0, b1, sigma, X1_1, X1_2, Y1, Y2


def load_traces(model_ids):
    # Load traces
    traces = {}
    for _id in model_ids:
        filepath = TMP_PATH.joinpath(f"az_trace_n{_id}.pkl")
        print(f"Loading {filepath}")
        with open(filepath, "rb") as f:
            traces[f"ncmp={_id}"] = pickle.load(f)
    return traces


def plot_traces(model_ids):

    traces = load_traces(model_ids)
    # extra = ["p", "category"]
    # , var_names=["b0", "b1", "sigma"]
    for key, trace in traces.items():
        axs = az.plot_trace(trace)
        fig = axs[0, 0].get_figure()
        fig.tight_layout()
        fig.savefig(TMP_PATH.joinpath(f"trace_{key}.png"))


def plot_posterior(model_ids):
    point_estimate = "mean"
    traces = load_traces(model_ids)
    for key, trace in traces.items():
        try:
            axs = az.plot_posterior(
                trace,
                var_names=["b0", "b1", "p", "sigma"],
                point_estimate=point_estimate,
            )
        except KeyError:
            axs = az.plot_posterior(
                trace, var_names=["b0", "b1"], point_estimate=point_estimate
            )

        fig = axs[0, 0].get_figure()
        fig.tight_layout()
        fig.savefig(TMP_PATH.joinpath(f"posterior_{key}.png"))


def sign(x):
    if x > 0:
        return "+"
    else:
        return "-"


def compare(model_ids, ics=["loo", "waic"]):
    """
    Load arviz traces from disk and compare
    """

    traces = load_traces(model_ids)

    # Analyze for each specified ic
    for ic in ics:
        print(ic)
        df_compare = az.compare(traces, ic=ic)
        print(df_compare)

        ax = az.plot_compare(df_compare, insample_dev=False)
        fig = ax.get_figure()
        fig.savefig(TMP_PATH.joinpath(f"compare_{ic}.png"))
        with open(TMP_PATH.joinpath(f"compare_{ic}.html"), "w") as f:
            f.write(df_compare.to_html())


def plot_data(xs, ys):
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    ax.set_title("Data")
    for x, y in zip(xs, ys):
        ax.scatter(
            x,
            y,
            s=144,
            marker="o",
            color="gray",
        )
    return fig


def plot_true_model(xs, ys, b0, b1, sigma):
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    x_model = np.linspace(-3, 3)
    ax.set_title("Data and true models")

    colors = itertools.cycle(get_cmap(CMAP_NAME).colors)
    for x, y in zip(xs, ys):
        color = next(colors)
        ax.scatter(
            x,
            y,
            s=144,
            marker="o",
            color=color,
        )

    colors = itertools.cycle(get_cmap(CMAP_NAME).colors)
    for icmp, b1_ in enumerate(b1):
        color = next(colors)
        y_model = b0 + b1_ * x_model
        ax.plot(
            x_model,
            y_model,
            linestyle="-",
            color=color,
            label=f"Cmp {icmp}: {b0} {sign(b1_)} {np.abs(b1_)}x + N(0,{sigma})",
        )
    ax.legend(fontsize=20, title="Y = b0 + b1*x + N(0, s)")
    return fig


# %% Make data

X1, Y, b0, b1, sigma, X1_1, X1_2, Y1, Y2 = make_data()
size = X1.size

# %% Run model

n_components = 2
trace, model = fit(
    X1,
    Y,
    n_components,
    # favor_few_components=False,
    # p_min=None,
    favor_few_components=True,
    nsteps=20000,
    return_inferencedata=True,
    save_trace=True,
)

# pm.model_to_graphviz(model)

# %% compare
model_ids = [1, 2, 3, 4]
compare(model_ids)

# %% Plot traces & posteriors
# print(trace)
# print(trace.sample_stats)
# https://python.arviz.org/en/stable/getting_started/WorkingWithInferenceData.html
# post = trace.posterior
# print(post["beta_1"])
# print(post.mean(dim=['chain', 'draw']))
# print(post.mean(dim=["draw"]))
# trace.sel(chain=[0, 2])
# stacked = az.extract(trace)
# idata.sel(draw=slice(100, None))


from pretty_html_table import build_table

model_ids = [2]
# plot_traces(model_ids)
plot_posterior(model_ids)

traces = load_traces(model_ids)
for model_id, data in traces.items():
    try:
        df = az.summary(traces[model_id], var_names=["b0", "b1", "p", "sigma"])
    except KeyError:
        df = az.summary(traces[model_id], var_names=["b0", "b1"])
    print(df)
    with open(TMP_PATH.joinpath(f"summary_{model_id}.html"), "w") as f:
        f.write(build_table(df, "blue_light", index=True))


# %% Plotting: checks

fig = plot_true_model((X1_1, X1_2), (Y1, Y2), b0, b1, sigma)
fig.savefig(RESULTS_PATH.joinpath("true_model.png"))
fig = plot_data((X1_1, X1_2), (Y1, Y2))
fig.savefig(RESULTS_PATH.joinpath("anonymous_data.png"))
fig = plot_curve_fit((X1_1, X1_2), (Y1, Y2))
fig.savefig(RESULTS_PATH.joinpath("curve_fit.png"))

# %% Bayesian
do_ppc = True
fig, fig_pair, fig_gppc, fig_mppc = plot_bayesian_fit(
    (X1_1, X1_2),
    (Y1, Y2),
    nsteps=10000,
    do_ppc=do_ppc,
)
fig.savefig(RESULTS_PATH.joinpath("bayesian_posterior.png"))
fig_pair.savefig(RESULTS_PATH.joinpath("bayesian_pair.png"))
if do_ppc:
    fig_gppc.savefig(RESULTS_PATH.joinpath("bayesian_gppc.png"))
    fig_mppc.savefig(RESULTS_PATH.joinpath("bayesian_mppc.png"))


# %% sklearn

# n_components = 3
# clsfier = GaussianMixture(
#     n_components=n_components,
#     random_state=0,
# )

n_components_gmm = 6
clsfier = BayesianGaussianMixture(
    n_components=n_components,
    random_state=42,
    weight_concentration_prior_type="dirichlet_distribution",
    weight_concentration_prior=1.0 / n_components_gmm,
)


#  (n_samples, n_features)
XY = np.stack((X1, Y), axis=1)
labels = clsfier.fit_predict(XY)

# %% Plotting: results


fig, axs = plt.subplots(3, 1, figsize=(20, 30), sharex=True)
x_model = np.linspace(-3, 3)
ax = axs[0]
ax.set_title("Data and underlying models components")
ax.scatter(
    X1,
    Y,
    s=144,
    marker="o",
    edgecolor="gray",
    facecolor="gray",
)
for icmp, b1_ in enumerate(b1):
    y_model = b0 + b1_ * x_model
    ax.plot(
        x_model,
        y_model,
        linestyle="-",
        color="gray",
        label=f"Cmp {icmp}: {b0} {sign(b1_)} {np.abs(b1_)}x + N(0,{sigma})",
    )
ax.legend(fontsize=20)

ax = axs[2]
ax.set_title(f"Linear Mixture Model with {n_components} components (pymc3)")
if n_components == 1:
    p_cat = np.ones(size, dtype="float")
    cat = np.zeros(size, dtype="int")
    categories = sorted(np.unique(cat))
    category_trace = [0] * len(trace["b0"])
    b1_trace = np.atleast_2d(trace["b1"])
# elif n_components == 2:
#     # this mean averages "how many times in the trace
#     # a point is categorized as either 0 or 1"
#     # the mean of the mean should give some center value that can be used to categorize
#     avg_cat = np.apply_along_axis(np.mean, 0, trace["category"])
#     p_cat = np.abs(avg_cat - 0.5) / 0.5
#     cat = avg_cat - np.mean(avg_cat) > 0
#     cat = cat.astype(int)
#     categories = sorted(np.unique(cat))
#     category_trace = trace["category"]
#     b1_trace = np.transpose(trace["b1"])
elif n_components >= 2:
    p_cat = trace["p"]
    print(p_cat.shape)
    print(p_cat)
    print(np.mean(p_cat, axis=0))
    print(np.mean(trace["b1"], axis=0))

    cat = trace["category"]
    print("cat", cat.shape)
    categories = sorted(np.unique(cat))
    # for eaach point count number of times it's classed (nclasses x npoints)
    a = np.apply_along_axis(np.bincount, 0, cat, minlength=np.max(cat) + 1)
    print(a.shape)
    print(a)
    print(np.sum(a, axis=1) / np.sum(a))

    # most prevalent cluster
    cat = np.argmax(a, axis=0)

    # Number of samples in most prevalent cluster
    n_cat = [a[c, i] for i, c in enumerate(cat)]

    # total number of samples
    n_tot = np.sum(a, axis=0)

    p_cat = n_cat / n_tot
    b1_trace = np.transpose(trace["b1"])

for icat in categories:

    # Calculate model mean
    y_model = b0_mean + b1_mean[icat] * x_model

    # Plot model mean
    (l,) = ax.plot(
        x_model,
        y_model,
        "-",
        label=f"Cmp {icat}: {b0_mean:7.2e} {sign(b1_mean[icat])} {np.abs(b1_mean[icat]):7.2e} * x )",
    )

    # Get the color
    color = l.get_color()
    color_rgba = mcolors.to_rgba(color)

    cat_count = np.sum(cat == icat)
    if cat_count == 0:
        continue

    # Expand to the number of data points in the class
    color_rgba = np.array([list(color_rgba) for p in p_cat[cat == icat]])

    # Set face alpha to the class probability
    color_rgba[:, 3] = p_cat[cat == icat]

    # Plot data points
    ax.scatter(
        X1[cat == icat],
        Y[cat == icat],
        s=144,
        marker="o",
        edgecolor=color,
        facecolor=color_rgba,
    )
ax.legend(fontsize=20, title="Mean parameters")

ax = axs[1]
colors = ["blue", "red", "magenta", "cyan", "green", "orange"]
means = clsfier.means_
covs = clsfier.covariances_
label_ids = set(labels)

ax.set_title(f"BayesianGaussianMixture with {n_components_gmm} initial components")
ax.scatter(
    X1,
    Y,
    s=144,
    marker="o",
    edgecolor="gray",
    facecolor="gray",
)

for mean, cov, color, label in zip(means, covs, colors, label_ids):
    plot_cov_ellipse(cov, mean, nstd=2, ax=ax, color=color, zorder=-1000, alpha=0.3)


ax.set_ylabel("y")
ax.set_xlabel("x1")
fig.savefig("regmix-classes.png")

print(pm.__version__)
