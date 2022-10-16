"""
Attempt to classify data in term of which model created the data and learn the models.
Should also attempt to figure out how many models is best (AIC or BIC)
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from two_models import make_data, plot, model, plot_parameters

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


print(xs_all.shape)
data = np.transpose(np.vstack((xs_all, ys_all)))
print(data.shape)
np.random.shuffle(data)
nmodels = 2
sets = np.split(data, nmodels)
print(sets)


colors = "red", "blue"
ssrs = []
for iset, (set, color) in enumerate(zip(sets, colors)):
    plot(
        set[:, 0],
        set[:, 1],
        axs[0, 0],
        color=color,
        label=f"Set {iset}",
    )

    popt, _ = curve_fit(model, set[:, 0], set[:, 1])
    ys_all_pred = model(xs_all, *popt)
    ssr = (ys_all - ys_all_pred) ** 2
    print(ssr)
    ssrs.append(ssr)

axs[0, 1].plot(ssrs[0], ssrs[1], "o")
axs[0, 1].set_xlabel("SSR model 1")
axs[0, 1].set_ylabel("SSR model 2")

# Make clustering

fig.tight_layout()
fig.savefig("two-models-cls.png")
