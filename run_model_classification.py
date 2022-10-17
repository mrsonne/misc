"""
Attempt to classify data in term of which model created the data and learn the models.
Should also attempt to figure out how many models is best (AIC or BIC)

Resources:
    https://towardsdatascience.com/gaussian-mixture-models-explained-6986aaf5a95

n_models = n_features = n_components
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.mixture import GaussianMixture
from two_models import make_data, plot, model, plot_parameters, plot_cov_ellipse

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
# np.random.shuffle(data)
n_models = 2
sets = np.split(data, n_models)

colors = "red", "blue"
# Sum of squared residualts
rs = []
for iset, (set, color) in enumerate(zip(sets, colors)):
    plot(
        set[:, 0],
        set[:, 1],
        axs[0, 0],
        color=color,
        label=f"Set {iset}",
    )

    plot(
        set[:, 0],
        set[:, 1],
        axs[1, 0],
        color=color,
        label=f"Set {iset}",
    )

    plot(
        set[:, 0],
        set[:, 1],
        axs[1, 1],
        color=color,
        label=f"Set {iset}",
    )

    # Fit model for current set
    popt, _ = curve_fit(model, set[:, 0], set[:, 1])

    # Predict for all data point using current model
    ys_all_pred = model(xs_all, *popt)

    # Plot prediction on original data split correctly into the two models
    plot(
        xs_all,
        ys_all_pred,
        axs[0, 0],
        linewidth=1,
        color=color,
        label=f"Model: {popt}",
    )

    plot(
        xs_all,
        ys_all_pred,
        axs[1, 1],
        linewidth=1,
        color=color,
        label=f"Model real split: {popt}",
    )

    # Collect residual for all data point in current model
    r = (ys_all - ys_all_pred) ** 2
    # r = ys_all - ys_all_pred
    rs.append(r)

#  (n_samples, n_features)
X = np.stack(rs, axis=1)
print(X.shape)
gm = GaussianMixture(n_components=n_models, random_state=0)
labels = gm.fit_predict(X)
label_ids = range(n_models)

proba = gm.predict_proba(X)
print(proba)


# shape: (n_components, n_features)
means = gm.means_
covs = gm.covariances_

for mean, cov, color, label in zip(means, covs, colors, label_ids):

    # Plot GMM elipses
    axs[0, 1].plot(mean[0], mean[1], "+k", markersize=20)
    plot_cov_ellipse(
        cov, mean, nstd=2, ax=axs[0, 1], color=color, zorder=-1000, alpha=0.3
    )

    # Plot GMM input data colored according to predicted classes
    axs[0, 1].plot(rs[0][labels == label], rs[1][labels == label], "o", color=color)

    # Plot original data with label color
    xs_ = xs_all[labels == label]
    ys_ = ys_all[labels == label]
    popt, _ = curve_fit(model, xs_, ys_)

    axs[1, 0].plot(
        xs_,
        ys_,
        "o",
        markersize=15,
        alpha=0.2,
        color=color,
        label=f"Predicted label: {label}",
    )

    ys_pred = model(xs_, *popt)

    # Plot prediction on original data split according to predicted classes
    axs[1, 1].plot(
        xs_,
        ys_pred,
        linestyle="dashed",
        linewidth=1,
        color=color,
        label=f"Model cls split: {popt}",
    )

    axs[1, 0].plot(
        xs_,
        ys_pred,
        linestyle="dashed",
        linewidth=1,
        color=color,
        label=f"Model: {popt}",
    )


axs[0, 1].set_xlabel("R model 1")
axs[0, 1].set_ylabel("R model 2")

for ax in axs.flat:
    ax.legend()

# Make clustering

fig.tight_layout()
fig.savefig("two-models-cls.png", dpi=180)
