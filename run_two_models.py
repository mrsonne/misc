import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from two_models import make_data, plot, model, plot_parameters

# https://docs.pymc.io/en/v3/pymc-examples/examples/generalized_linear_models/GLM-linear.html

np.random.seed(0)
irow_1_model = 0
irow_2_models = 1
xs_model1, ys_model1, xs_model2, ys_model2, xs_all, ys_all = make_data()

fig, axs = plt.subplots(2, 2, figsize=(16, 16))
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

ymins, ymaxs = zip(*[ax.get_ylim() for ax in axs[:, 1]])
ymin = min(ymins)
ymax = max(ymaxs)
[ax.set_ylim((ymin, ymax)) for ax in axs[:, 1]]
xmins, xmaxs = zip(*[ax.get_xlim() for ax in axs[:, 1]])
xmin = min(xmins)
xmax = max(xmaxs)
[ax.set_xlim((xmin, xmax)) for ax in axs[:, 1]]

fig.tight_layout()
fig.savefig("two-models.png")
