# %% Imports

# https://stats.stackexchange.com/questions/185812/regression-mixture-in-pymc3


##Fake data
import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm

# import seaborn as sns

# %% Data

np.random.seed(123)
alpha = 0
sigma = 1
beta = [-5]
beta2 = [5]
# size = 250
size = 50

# Predictor variable
# X1_1 = np.random.randn(size)
X1_1 = np.linspace(-2, 2, size)

# Simulate outcome variable--cluster 1
Y1 = alpha + beta[0] * X1_1 + np.random.normal(loc=0, scale=sigma, size=size)

# Predictor variable
# X1_2 = np.random.randn(size)
X1_2 = np.linspace(-2, 2, size)
# Simulate outcome variable --cluster 2
Y2 = alpha + beta2[0] * X1_2 + np.random.normal(loc=0, scale=sigma, size=size)


X1 = np.append(X1_1, X1_2)
Y = np.append(Y1, Y2)


# %% Model

basic_model = pm.Model()

with basic_model:
    p = pm.Uniform("p", 0, 1)  # Proportion in each mixture

    alpha = pm.Normal("alpha", mu=0, sd=10)  # Intercept
    beta_1 = pm.Normal("beta_1", mu=0, sd=100, shape=2)  # Betas.  Two of them.
    sigma = pm.Uniform("sigma", 0, 20)  # Noise

    category = pm.Bernoulli(
        "category", p=p, shape=size * 2
    )  # Classification of each observation

    b1 = pm.Deterministic("b1", beta_1[category])  # Choose beta based on category

    mu = alpha + b1 * X1  # Expected value of outcome

    # Likelihood
    Y_obs = pm.Normal("Y_obs", mu=mu, sd=sigma, observed=Y)


with basic_model:
    step1 = pm.Metropolis([p, alpha, beta_1, sigma])
    # non-default scaling is importtant
    step2 = pm.BinaryMetropolis([category], scaling=0.01)
    trace = pm.sample(20000, [step1, step2], progressbar=True)


alpha_mean = np.apply_along_axis(np.mean, 0, trace["alpha"])
print(alpha_mean)


# %% Plotting: checks

import arviz as az

# Gives errors
axs = az.plot_trace(trace)
fig = axs[0, 0].get_figure()
fig.savefig("regmix-trace.png")

# %% Plotting: results

# this mean averages "how many times in the trace 
# a point is averaged as either 0 or 1"
# the mean of the mean should give some center value that can be used to categorize
p_cat = np.apply_along_axis(np.mean, 0, trace["category"])
cat = p_cat - np.mean(p_cat) < 0

# print(p_cat)
fig, ax = plt.subplots(1, 1, figsize=(10, 4))
ax.scatter(X1, Y, c=p_cat, cmap="coolwarm")
ax.set_ylabel("Y")
ax.set_xlabel("X1")

x_model = np.linspace(-3, 3)
y1_model = alpha_mean + beta_1_mean[0] * x_model
y2_model = alpha_mean + beta_1_mean[1] * x_model
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
counts, bins = np.histogram(trace["alpha"], bins=50)
fig, ax = plt.subplots(1, 1)
ax.hist(bins[:-1], bins, weights=counts)
ax.set_ylabel("count")
ax.set_xlabel("alpha")
fig.savefig("regmix-alpha-posterior.png")


# %% plot posterior for beta
fig, ax = plt.subplots(1, 1)
ax.plot(trace["beta_1"][:, 0])
ax.plot(trace["beta_1"][:, 1])
fig.savefig("regmix-beta-trace.png")

counts, bins = np.histogram(trace["beta_1"][:, 1], bins=50)
beta_1_mean = np.apply_along_axis(np.mean, 0, trace["beta_1"])
print(beta_1_mean)
fig, ax = plt.subplots(1, 1)
ax.hist(bins[:-1], bins, weights=counts)
ax.set_ylabel("count")
ax.set_xlabel("beta_1")
fig.savefig("regmix-beta1-posterior.png")


# %%
# trace.sel(draw=slice(1000, None))
print(trace["category"].shape)
category_trace = trace["category"]
# get the mean models
beta1_trace = trace["beta_1"]
beta1_trace = beta1_trace[:,0]
# beta_1_mean = np.apply_along_axis(np.mean, 0, beta1_trace)
# print(beta_1_mean)
print(beta1_trace.shape
beta_1_mean  = beta1_trace[category_trace == 0].mean()
print(beta_1_mean)


print(pm.__version__)
