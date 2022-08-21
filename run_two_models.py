import numpy as np
from two_models import generate_data, plot
import matplotlib.pyplot as plt

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

fig, ax = plt.subplots()
plot(xs_model1, ys_model1, ax, marker="x", color="blue")
plot(xs_model2, ys_model2, ax, marker="x", color="magenta")
fig.savefig("two-models.png")
