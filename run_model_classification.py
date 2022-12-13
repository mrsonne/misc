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
from sklearn.cluster import KMeans, DBSCAN
from two_models import make_data, plot, model, plot_parameters, plot_cov_ellipse

np.random.seed(0)
xs_model1, ys_model1, xs_model2, ys_model2, xs_all, ys_all = make_data()

fig, axs = plt.subplots(2, 3, figsize=(24, 16))
colors = "red", "blue"

axs[0, 0].plot(xs_all, ys_all, "ok")
axs[0, 0].set_title("All data")
axs[0, 1].set_title("All data divided in true groups. Model fit in each group.")
axs[0, 2].set_title("All data in R-space. GMM classes + components.")
axs[1, 0].set_title("All data in true group. GMM prediction + log-likelihood.")
axs[1, 1].set_title(
    "All data divided in groups (true & predicted). Model fit in each predicted group."
)

axs[1, 1].set_title(
    "All data divided in true groups. Model fit in each true & predicted group."
)


plot(
    xs_model1,
    ys_model1,
    axs[0, 1],
    marker="x",
    color=colors[0],
    label=f"Data model A",
)
plot(
    xs_model2,
    ys_model2,
    axs[0, 1],
    marker="x",
    color=colors[1],
    label=f"Data model B",
)


print(xs_all.shape)
data = np.transpose(np.vstack((xs_all, ys_all)))
print(data.shape)
# np.random.shuffle(data)
n_models = 2
sets = np.split(data, n_models)

# Sum of squared residualts
rs = []
for iset, (set, color) in enumerate(zip(sets, colors)):
    plot(
        set[:, 0],
        set[:, 1],
        axs[0, 1],
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

    plot(
        set[:, 0],
        set[:, 1],
        axs[1, 2],
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
        axs[0, 1],
        linewidth=1,
        color=color,
        label=f"Model for set {popt}",
        marker=None,
    )

    plot(
        xs_all,
        ys_all_pred,
        axs[1, 2],
        linewidth=1,
        color=color,
        label=f"Model real split: {popt}",
        marker=None,
    )

    # Collect residual for all data point in current model
    r = (ys_all - ys_all_pred) ** 2
    sigma_sq = np.mean(r)
    # r = r / (2 * sigma_sq)  # - len(ys_all) / 2 * np.log(sigma_sq)
    r = r / sigma_sq
    # r = ys_all - ys_all_pred
    rs.append(r)

#  (n_samples, n_features)
X = np.stack(rs, axis=1)
print(X.shape)

# GMM
covariance_type = "diag"
clsfier = GaussianMixture(
    n_components=n_models,
    random_state=0,
    covariance_type=covariance_type,
    reg_covar=1e-3,
)

# clsfier = KMeans(n_clusters=n_models, random_state=0)
# clsfier = DBSCAN(eps=5, min_samples=10)

labels = clsfier.fit_predict(X)
print(labels)

label_ids = range(n_models)


plot(
    xs_model1,
    ys_model1,
    axs[1, 0],
    marker="x",
    color=colors[0],
    label=f"Data model A",
)
plot(
    xs_model2,
    ys_model2,
    axs[1, 0],
    marker="x",
    color=colors[1],
    label=f"Data model A",
)


if isinstance(clsfier, GaussianMixture):
    proba = clsfier.predict_proba(X)
    size = 500 * proba[:, 0]
    axs[1, 0].scatter(
        xs_all,
        ys_all,
        s=size,
        facecolors="None",
        edgecolors=colors[0],
        label="log-likelihood model 1",
    )
    size = 500 * proba[:, 1]
    axs[1, 0].scatter(
        xs_all,
        ys_all,
        s=size,
        facecolors="None",
        edgecolors=colors[1],
        label="log-likelihood model 2",
    )

    # shape: (n_components, n_features)
    means = clsfier.means_
    covs = clsfier.covariances_

    for mean, cov, color, label in zip(means, covs, colors, label_ids):
        if covariance_type in ["diag"]:
            cov_ = np.zeros((n_models, n_models))
            np.fill_diagonal(cov_, cov)
            print(cov_)
        elif covariance_type == "tied":
            cov_ = covs
        else:
            cov_ = cov

        # Plot GMM elipses
        axs[0, 1].plot(mean[0], mean[1], "+k", markersize=20)
        plot_cov_ellipse(
            cov_, mean, nstd=2, ax=axs[0, 2], color=color, zorder=-1000, alpha=0.3
        )

for color, label in zip(colors, label_ids):

    # Plot GMM input data colored according to predicted classes
    axs[0, 2].plot(rs[0][labels == label], rs[1][labels == label], "o", color=color)

    # Plot original data with label color
    xs_ = xs_all[labels == label]
    ys_ = ys_all[labels == label]
    xs_, ys_ = zip(*sorted(zip(xs_, ys_)))
    xs_ = np.array(xs_)
    ys_ = np.array(ys_)
    popt, _ = curve_fit(model, xs_, ys_)

    axs[1, 1].plot(
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
    axs[1, 2].plot(
        xs_,
        ys_pred,
        linestyle="dashed",
        linewidth=1,
        color=color,
        label=f"Model cls split: {popt}",
    )

    axs[1, 1].plot(
        xs_,
        ys_pred,
        linestyle="dashed",
        linewidth=1,
        color=color,
        label=f"Model: {popt}",
    )


axs[0, 2].set_xlabel("R in model 1")
axs[0, 2].set_ylabel("R in model 2")

for ax in axs.flat:
    ax.legend()

# Make clustering

fig.tight_layout()
fig.savefig("two-models-cls.png", dpi=180)
