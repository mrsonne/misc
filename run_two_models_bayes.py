import arviz as az
from pymc3 import Model, HalfCauchy, Normal, sample

# Generate data
x = [1, 2, 3, 4, 5, 6]
y = [1.1, 1.9, 3.0, 4.2, 4.9, 6.14]

# Infer parameters
nsample = 50
with Model() as model:
    sigma = HalfCauchy("sigma", beta=10, testval=1.0)
    intercept = Normal("intercept", 0, sigma=20)
    x_coeff = Normal("x", 0, sigma=20)

    likelihood = Normal("y", mu=intercept + x_coeff * x, sigma=sigma, observed=y)
    trace = sample(nsample, return_inferencedata=True)

# Plot
axs = az.plot_trace(trace, figsize=(10, 7))
fig = axs[0, 0].get_figure()
fig.savefig("trace.png")
