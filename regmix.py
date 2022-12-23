# https://stats.stackexchange.com/questions/185812/regression-mixture-in-pymc3


##Fake data
import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm

# import seaborn as sns

#  Data
np.random.seed(123)
alpha = 0
sigma = 1
beta = [-5]
beta2 = [5]
# size = 250
size = 100

# Predictor variable
X1_1 = np.random.randn(size)

# Simulate outcome variable--cluster 1
Y1 = alpha + beta[0] * X1_1 + np.random.normal(loc=0, scale=sigma, size=size)

# Predictor variable
X1_2 = np.random.randn(size)
# Simulate outcome variable --cluster 2
Y2 = alpha + beta2[0] * X1_2 + np.random.normal(loc=0, scale=sigma, size=size)


X1 = np.append(X1_1, X1_2)
Y = np.append(Y1, Y2)


#  Model

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

# with basic_model:
#     step1 = pm.Metropolis([p, alpha, beta_1, sigma])
#     step2 = pm.BinaryMetropolis([category])
#     trace = pm.sample(20000, [step1, step2], progressbar=True)

with basic_model:
    step1 = pm.Metropolis([p, alpha, beta_1, sigma])
    step2 = pm.BinaryMetropolis([category], scaling=0.01)
    trace = pm.sample(100000, [step1, step2], progressbar=True)

# pm.traceplot(trace)

#  Plotting


p_cat = np.apply_along_axis(np.mean, 0, trace["category"])
print(p_cat)
fig, axes = plt.subplots(1, 1, figsize=(10, 4))
axes.scatter(X1, Y, c=p_cat, cmap="coolwarm")

axes.set_ylabel("Y")
axes.set_xlabel("X1")

fig.savefig("regmix.png")
