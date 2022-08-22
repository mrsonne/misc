from random import uniform
import arviz as az
import pymc3 as pm
import seaborn
import matplotlib.pyplot as plt
from two_models import make_data

# Generate data
_, _, _, _, xs_all, ys_all = make_data()

# Infer parameters
# nsample = 500
# nchains = 2
nsample = 5000
nchains = 16
with pm.Model() as model:
    b0 = pm.Normal("b0", 0, sigma=20)
    b1 = pm.Normal("b1", 0, sigma=20)
    y_est = b0 + b1 * xs_all
    sigma = pm.HalfCauchy("sigma", beta=10, testval=1.0)
    likelihood = pm.Normal("y", mu=y_est, sigma=sigma, observed=ys_all)

    # # define prior for StudentT degrees of freedom
    # # InverseGamma has nice properties:
    # # it's continuous and has support x ∈ (0, inf)
    # nu = pm.InverseGamma("nu", alpha=1, beta=1)
    # # define Student T likelihood
    # likelihood = pm.StudentT("likelihood", mu=y_est, sigma=2, nu=nu, observed=ys_all)

    trace = pm.sample(nsample, return_inferencedata=True, chains=nchains)

print(type(trace))
# Plot
axs = az.plot_trace(trace, figsize=(16, 12))
fig = axs[0, 0].get_figure()
fig.savefig("trace.png")

# print(trace.posterior["intercept"].stack())

plt.figure(figsize=(9, 7))
print(trace)
print(trace.posterior.mean())
stacked = az.extract_dataset(trace)
print("HEJ")
p = seaborn.jointplot(
    x=stacked.b1.values, y=stacked.b0.values, kind="hex", color="#4CB391"
)
fig = p.fig
axs = fig.axes
fig.suptitle(trace.posterior.mean())
# p.ax_joint.collections[0].set_alpha(0)

axs[0].set_xlabel("b1")
axs[0].set_ylabel("b0")
fig.subplots_adjust(top=0.95, bottom=0.0)  # Reduce plot to make room
fig.tight_layout()
fig.savefig("heat.png")
