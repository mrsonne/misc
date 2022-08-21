import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from two_models import generate_data, plot, model, plot_parameters

np.random.seed(0)
sigma = 0.5
irow_1_model = 0
irow_2_models = 1
print(f"sigma {sigma}")

xmin, xmax = -1, 1
pars_model1 = 1.0, 1.0
sigma_model1 = sigma
n_model1 = 10
xs_model1 = np.linspace(xmin, xmax, n_model1)
ys_model1 = generate_data(xs_model1, pars_model1, sigma_model1)

pars_model2 = -1.0, 1.0
sigma_model2 = sigma
n_model2 = 10
xs_model2 = np.linspace(xmin, xmax, n_model2)
ys_model2 = generate_data(xs_model2, pars_model2, sigma_model2)

xs_all = np.concatenate((xs_model1, xs_model2))
ys_all = np.concatenate((ys_model1, ys_model2))

fig, axs = plt.subplots(2, 2, figsize=(16, 12), sharex=True)
plot(
    xs_model1,
    ys_model1,
    axs[irow_2_models, 0],
    marker="x",
    color="blue",
    label=f"Data model 1",
)
plot(
    xs_model2,
    ys_model2,
    axs[irow_2_models, 0],
    marker="x",
    color="magenta",
    label=f"Data model 2",
)

plot(
    xs_all,
    ys_all,
    axs[irow_1_model, 0],
    marker="o",
    color="red",
    label=f"Data models 1 & 2",
)

# Fit the data
popt_model1, pcov1 = curve_fit(model, xs_model1, ys_model1)
popt_model2, pcov2 = curve_fit(model, xs_model2, ys_model2)
popt_all, pcov_all = curve_fit(model, xs_all, ys_all)

print(popt_model1, pcov1)
print(popt_model2, pcov2)
print(popt_all, pcov_all)

plot_parameters(popt_model1, pcov1, axs[irow_2_models, 1], "Data model 1")
plot_parameters(popt_model2, pcov2, axs[irow_2_models, 1], "Data model 2")
plot_parameters(popt_all, pcov_all, axs[irow_1_model, 1], "Data models 1 & 2")


ys_predicted_model1 = model(xs_model1, *popt_model1)
ys_predicted_model2 = model(xs_model2, *popt_model2)
ys_predicted_all = model(xs_all, *popt_all)

plot(
    xs_model1,
    ys_predicted_model1,
    axs[irow_2_models, 0],
    marker="",
    linewidth=1,
    color="blue",
    label=f"Model 1",
)
plot(
    xs_model2,
    ys_predicted_model2,
    axs[irow_2_models, 0],
    marker="",
    linewidth=1,
    color="magenta",
    label=f"Model 2",
)

plot(
    xs_all,
    ys_predicted_all,
    axs[irow_1_model, 0],
    marker="",
    linewidth=1,
    color="red",
    label=f"Model",
)

for ax in axs.flat:
    ax.legend()

fig.savefig("two-models.png")
