import arviz as az
from pymc3 import Model, HalfCauchy, Normal, sample
from two_models import make_data

# Generate data
_, _, _, _, xs_all, ys_all = make_data()

# Infer parameters
nsample = 1000
with Model() as model:
    sigma = HalfCauchy("sigma", beta=10, testval=1.0)
    intercept = Normal("intercept", 0, sigma=20)
    x_coeff = Normal("x", 0, sigma=20)

    likelihood = Normal(
        "y", mu=intercept + x_coeff * xs_all, sigma=sigma, observed=ys_all
    )
    trace = sample(nsample, return_inferencedata=True, chains=10)

# Plot
axs = az.plot_trace(trace, figsize=(16, 12))
fig = axs[0, 0].get_figure()
fig.savefig("trace.png")
