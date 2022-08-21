import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from two_models import generate_data, plot, model

np.random.seed(23)

xmin, xmax = -1, 1
pars_model1 = 1.0, 1.0
sigma_model1 = 0.1
n_model1 = 10
xs_model1 = np.linspace(xmin, xmax, n_model1)
ys_model1 = generate_data(xs_model1, pars_model1, sigma_model1)

pars_model2 = -1.0, 1.0
sigma_model2 = 0.1
n_model2 = 10
xs_model2 = np.linspace(xmin, xmax, n_model2)
ys_model2 = generate_data(xs_model2, pars_model2, sigma_model2)

xs_all = np.concatenate((xs_model1, xs_model2))
ys_all = np.concatenate((ys_model1, ys_model2))

fig, axs = plt.subplots(2, 2, figsize=(16, 12), sharex=True)
plot(xs_model1, ys_model1, axs[0, 0], marker="x", color="blue", label=f"Data model 1")
plot(
    xs_model2, ys_model2, axs[0, 0], marker="x", color="magenta", label=f"Data model 2"
)

plot(xs_all, ys_all, axs[1, 0], marker="o", color="red", label=f"Data models 1 & 2")

# Fit the data
popt_model1, pcov1 = curve_fit(model, xs_model1, ys_model1)
popt_model2, pcov2 = curve_fit(model, xs_model2, ys_model2)
popt_all, pcov_all = curve_fit(model, xs_all, ys_all)

print(popt_model1, pcov1)
print(popt_model1, pcov2)
print(popt_all, pcov_all)

ys_predicted_model1 = model(xs_model1, *popt_model1)
ys_predicted_model2 = model(xs_model2, *popt_model2)
ys_predicted_all = model(xs_all, *popt_all)

plot(
    xs_model1,
    ys_predicted_model1,
    axs[0, 0],
    marker="",
    linewidth=1,
    color="blue",
    label=f"model 1 {popt_model1}",
)
plot(
    xs_model2,
    ys_predicted_model2,
    axs[0, 0],
    marker="",
    linewidth=1,
    color="magenta",
    label=f"model 2 {popt_model2}",
)

plot(
    xs_all,
    ys_predicted_all,
    axs[1, 0],
    marker="",
    linewidth=1,
    color="red",
    label=f"model {popt_all}",
)

for ax in axs.flat:
    ax.legend()

fig.savefig("two-models.png")
