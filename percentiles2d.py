# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 12:24:53 2018

@author: SonneJ
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, multivariate_normal, chi2
from matplotlib.patches import Ellipse
np.random.seed(0)

def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the 
    ellipse patch artist.

    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the 
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.

    Returns
    -------
        A matplotlib ellipse artist
    """
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)
    ax.add_patch(ellip)
    return ellip


def pdf_kde(q, w, nresample, pars):
    """
    """
    # approximate the pdf
    pdf = gaussian_kde(pars['xy'].T)
    # densities on the specified grid
    densities_on_grid = pdf([q.flatten(), w.flatten()])
    # sample the pdf
    smpl = pdf.resample(nresample)
    # probability densities represented both by the elements values and the number of elements
    densities_resampled = pdf(smpl)
    return densities_on_grid, densities_resampled


def pdf_normal(q, w, nresample, pars):   
    """
    See 'pdf_kde'
    """
    densities_on_grid = multivariate_normal.pdf(np.array([q.flatten(), w.flatten()]).T,
                                                mean=pars['mean'],
                                                cov=pars['cov'])
                     
    smpl = np.random.multivariate_normal(pars['mean'],
                                         pars['cov'],
                                         nresample)

    densities_resampled = multivariate_normal.pdf(smpl,
                                                     mean=pars['mean'],
                                                     cov=pars['cov'])
    return densities_on_grid, densities_resampled


def percentile_region(percentiles, q, w, nresample, density_estimator, pars):
    """ https://stats.stackexchange.com/questions/64680/how-to-determine-quantiles-isolines-of-a-multivariate-normal-distribution?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
    """
    qs = 100. - percentiles
    densities_on_grid, densities_resampled = density_estimator(q, w, nresample, pars)
    density_levels_for_percentiles = np.percentile(densities_resampled, qs)
    densities_on_grid.shape = q.shape
    return densities_on_grid, density_levels_for_percentiles


def percentiles_to_std(percentiles, df):
    """critical chi^2 values used for converting standard 
    deviation to percentile

    http://www.real-statistics.com/multivariate-statistics/multivariate-normal-distribution/confidence-hyper-ellipse-eigenvalues/
    """
    chi2crits = [chi2.ppf(0.01*pct, df) for pct in percentiles]
    return np.sqrt(chi2crits)


def run():
    """
    """
    # Parameters for generating synthetic data
    ndata = 2500
    x_mean = 100
    y_mean = 400
    cov = np.array([[120,80],[80,80]])

    # generate synthetic data
    xy = np.random.multivariate_normal((x_mean, y_mean), cov, ndata)
    xs = xy[:,0]; ys = xy[:,1]

    # estimate confidence region using these models for the data
    titles = '"KDE"', '"bivariate normal"'
    density_funcs = pdf_kde, pdf_normal
    density_pars = {'xy': xy}, {'mean':(x_mean, y_mean), 'cov':cov}

    nplts = len(titles)
    fig, axs = plt.subplots(nplts, 1, figsize=(6, nplts*6))
    axs = np.atleast_1d(axs)
    
    # percentiles expressed as standard deviations
    percentiles = 68.2, 95.
    df = 2
    nstd = percentiles_to_std(percentiles, df)

    colors = 'red', 'orange'
    percentiles, colors, nstd = zip(*sorted(zip(percentiles, colors, nstd), reverse=True))
    percentiles = np.array(percentiles)

    min_x, max_x  = np.min(xs), np.max(xs)
    pad_x = (max_x - min_x)*0.05
    min_y, max_y = np.min(ys), np.max(ys)
    pad_y = (max_y - min_y)*0.05

    # make a grid
    xtmp = np.linspace(min_x - pad_x, max_x + pad_x, 100) 
    ytmp = np.linspace(min_y - pad_y, max_y + pad_y, 200)
    q, w = np.meshgrid(xtmp, ytmp)

    # number of times to sample distribution to get the 
    # probability at the specified percentiles
    nresample = 1000
    
    points = np.stack((xs, ys), axis=-1)
    pos = points.mean(axis=0)
    
    for iplt, (_estimator, _pars) in enumerate(zip(density_funcs, density_pars)):
        axs[iplt].set_title('PDF modeled using {}'.format(titles[iplt]))
        densities_on_grid, density_levels_for_percentiles = percentile_region(percentiles,
                                                                              q, w, nresample,
                                                                              _estimator, _pars)

        cs = axs[iplt].contour(xtmp, ytmp, densities_on_grid, density_levels_for_percentiles, colors=colors)
        points_inside_all = np.zeros_like(xs)
        for iscore, (fact, color, pct, score) in enumerate(zip(nstd, colors, percentiles, density_levels_for_percentiles)):
            plot_cov_ellipse(cov, pos, nstd=fact, 
                             facecolor='none', 
                             edgecolor=color, ax=axs[iplt], 
                             linewidth=2, 
                             linestyle='dotted',)
            axs[iplt].plot([],[], linestyle=':', color=color, label='{} % (true)'.format(pct))
            collection = cs.collections[iscore]
            for path in collection.get_paths():
                points_inside = path.contains_points(points)
                points_inside_all += points_inside
                fraction = float(len(xs[points_inside]))/len(xs)
                axs[iplt].scatter(xs[points_inside],
                                  ys[points_inside],
                                  alpha=0.2,
                                  color=color,
                                  label='Data (fraction inside:{:4.2f})'.format(fraction))

    
            axs[iplt].plot([],[], linestyle='-', color=color, label='{} % (data & {})'.format(pct, titles[iplt]))

        points_outside = np.where(points_inside_all > 0, False, True)
        axs[iplt].scatter(xs[points_outside], ys[points_outside], alpha=0.2, color='gray', label='Data outside')
        handles, labels = axs[iplt].get_legend_handles_labels()
        axs[iplt].legend(handles[::-1], labels[::-1])

    plt.tight_layout()
    plt.savefig('percentiles2d.png')
    plt.show()

if __name__ == '__main__':
    run()