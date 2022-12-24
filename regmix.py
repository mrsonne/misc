# %% Imports


##Fake data
import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm

import theano.tensor as tt


def c1(X1, Y):
    """One-component "mixture" """
    nsample = 5000
    nchains = 16
    with pm.Model() as model:
        b0 = pm.Normal("b0", 0, sigma=20)
        b1 = pm.Normal("b1", 0, sigma=20)
        y_est = b0 + b1 * X1
        sigma = pm.HalfCauchy("sigma", beta=10, testval=1.0)
        likelihood = pm.Normal("y", mu=y_est, sigma=sigma, observed=Y)

        trace = pm.sample(nsample, return_inferencedata=True, chains=nchains)
        return trace


def c3(X1, Y):
    ...


def c2(X1, Y):
    """Two component model
    Modified from https://stats.stackexchange.com/questions/185812/regression-mixture-in-pymc3
    """
    n_components = 2
    size = X1.size
    basic_model = pm.Model()
    with basic_model:
        p = pm.Uniform("p", 0, 1)  # Proportion in each mixture

        b0 = pm.Normal("b0", mu=0, sd=10)  # Intercept
        b1 = pm.Normal("b1", mu=0, sd=100, shape=n_components)  # Betas.  Two of them.
        sigma = pm.Uniform("sigma", 0, 20)  # Noise

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
            return_inferencedata=False,
            start={"b1": np.linspace(-1, 1, n_components)},  #
        )

    return trace


# import seaborn as sns

# %% Data

np.random.seed(123)
b0 = 0
sigma = 2
b1 = [-5, 5]
# size = 250
size1 = 25
size2 = 35

# Predictor variable
# X1_1 = np.random.randn(size)
X1_1 = np.linspace(-2, 2, size1)

# Simulate outcome variable--cluster 1
Y1 = b0 + b1[0] * X1_1 + np.random.normal(loc=0, scale=sigma, size=size1)

# Predictor variable
# X1_2 = np.random.randn(size)
X1_2 = np.linspace(-2, 2, size2)
# Simulate outcome variable --cluster 2
Y2 = b0 + b1[1] * X1_2 + np.random.normal(loc=0, scale=sigma, size=size2)


X1 = np.append(X1_1, X1_2)
Y = np.append(Y1, Y2)


# %% Run model
trace = c2(X1, Y)

# %% post simulation calcs
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

print(trace["b0"].shape)
b0_mean = np.apply_along_axis(np.mean, 0, trace["b0"])
b1_mean = np.apply_along_axis(np.mean, 0, trace["b1"])
print(b0_mean)
print(b1_mean)


# %% Plotting: checks

import arviz as az

# Gives errors
axs = az.plot_trace(trace)
fig = axs[0, 0].get_figure()
fig.savefig("regmix-trace.png")

# %% Plotting: results

# this mean averages "how many times in the trace
# a point is categorized as either 0 or 1"
# the mean of the mean should give some center value that can be used to categorize
p_cat = np.apply_along_axis(np.mean, 0, trace["category"])
cat = p_cat - np.mean(p_cat) < 0

# print(p_cat)
fig, ax = plt.subplots(1, 1, figsize=(10, 4))
ax.scatter(X1, Y, c=p_cat, cmap="coolwarm")
ax.set_ylabel("Y")
ax.set_xlabel("X1")

x_model = np.linspace(-3, 3)
y1_model = b0_mean + b1_mean[0] * x_model
y2_model = b0_mean + b1_mean[1] * x_model
ax.plot(x_model, y1_model, "b-")
ax.plot(x_model, y2_model, "r-")
fig.savefig("regmix-classes.png")


# The probability seems not to be localized around 0 and 1
# It's not bimodal around 0.5 if the R^2 is small it seems
# In the comments for the original script it says
# "Proportion in each mixture"
counts, bins = np.histogram(p_cat, bins=50)
fig, ax = plt.subplots(1, 1)
ax.hist(bins[:-1], bins, weights=counts)
ax.set_ylabel("count")
ax.set_xlabel("p_cat")
ax.set_xlim((0, 1))
fig.savefig("regmix-p_cat-posterior.png")

# %% plot posterior
counts, bins = np.histogram(trace["b0"], bins=50)
fig, ax = plt.subplots(1, 1)
ax.hist(bins[:-1], bins, weights=counts)
ax.set_ylabel("count")
ax.set_xlabel("b0")
fig.savefig("regmix-b0-posterior.png")


# %% plot posterior for beta
fig, ax = plt.subplots(1, 1)
ax.plot(trace["b1"][:, 0])
ax.plot(trace["b1"][:, 1])
fig.savefig("regmix-b1-trace.png")

counts, bins = np.histogram(trace["b1"], bins=50)
b1_mean = np.apply_along_axis(np.mean, 0, trace["b1"])
print(b1_mean)
fig, ax = plt.subplots(1, 1)
ax.hist(bins[:-1], bins, weights=counts)
ax.set_ylabel("count")
ax.set_xlabel("b1")
fig.savefig("regmix-b1-posterior.png")


# %%
# trace.sel(draw=slice(1000, None))
print(trace["category"].shape)
category_trace = trace["category"]
# get the mean models
b1_trace = trace["b1"]
b1_trace = b1_trace[:, 0]
# beta_1_mean = np.apply_along_axis(np.mean, 0, beta1_trace)
# print(beta_1_mean)
print(b1_trace.shape)
# beta_1_mean  = beta1_trace[category_trace == 0].mean()
# print(beta_1_mean)


print(pm.__version__)

print(trace["b1_"])
fig, ax = plt.subplots(1, 1)
ax.plot(trace["b1_"][:, 0])
ax.plot(trace["b1_"][:, 1])
