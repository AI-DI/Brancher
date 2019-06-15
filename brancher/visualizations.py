import warnings

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi


def plot_posterior_histogram(model, variables, number_samples=300): #TODO: fix code duplication

    # Get samples
    sample = model.get_sample(number_samples)
    post_sample = model.get_posterior_sample(number_samples)

    # Join samples
    sample["Mode"] = "Prior"
    post_sample["Mode"] = "Posterior"
    subsample = sample[variables + ["Mode"]]
    post_subsample = post_sample[variables + ["Mode"]]
    joint_subsample = subsample.append(post_subsample)

    # Plot posterior
    warnings.filterwarnings('ignore')
    g = sns.PairGrid(joint_subsample, hue="Mode")
    g = g.map_offdiag(sns.distplot)
    g = g.map_diag(sns.distplot)
    g = g.add_legend()
    warnings.filterwarnings('default')


def plot_posterior(model, variables, number_samples=1000):

    # Get samples
    sample = model.get_sample(number_samples)
    post_sample = model.get_posterior_sample(number_samples)

    # Join samples
    sample["Mode"] = "Prior"
    post_sample["Mode"] = "Posterior"
    subsample = sample[variables + ["Mode"]]
    post_subsample = post_sample[variables + ["Mode"]]
    joint_subsample = subsample.append(post_subsample)

    # Plot posterior
    warnings.filterwarnings('ignore')
    g = sns.PairGrid(joint_subsample, hue="Mode")
    g = g.map_offdiag(sns.kdeplot)
    g = g.map_diag(sns.kdeplot, lw=3, shade=True)
    g = g.add_legend()
    warnings.filterwarnings('default')


def plot_density(model, variables, number_samples=2000):
    sample = model.get_sample(number_samples)
    warnings.filterwarnings('ignore')
    g = sns.PairGrid(sample[variables])
    g = g.map_offdiag(sns.kdeplot)
    g = g.map_diag(sns.kdeplot, lw=3, shade=True)
    g = g.add_legend()
    warnings.filterwarnings('default')


def ensemble_histogram(sample_list, variable, weights, bins=30):
    num_samples = sum([len(s) for s in sample_list])
    num_resamples = [int(np.ceil(w*num_samples*2)) for w in weights]
    max_samples = max(num_resamples)
    hist_df = pd.DataFrame()
    for idx, s in enumerate(sample_list):
        num_remaining_samples = max_samples - num_resamples[idx]
        resampled_values = np.concatenate([s[variable].sample(num_resamples[idx], replace=True).values,
                                           np.array([np.nan]*num_remaining_samples)])
        hist_df["Model {}".format(idx)] = resampled_values
    hist_df.plot.hist(stacked=True, bins=bins)


def plot_particles(particles, var_name, var2_name=None, dim1=0, dim2=0, xlim=None, ylim=None, colors=None):
    if not var2_name:
        var2_name = var_name
    extracted_particles = np.transpose(np.array([[p.get_variable(name).value.detach().numpy().flatten()[dim] for p in particles]
                            for dim, name in [(dim1, var_name), (dim2, var2_name)]]))

    def voronoi_finite_polygons_2d(vor, radius=1000):
        """
        Reconstruct infinite voronoi regions in a 2D diagram to finite
        regions.

        Parameters
        ----------
        vor : Voronoi
            Input diagram
        radius : float, optional
            Distance to 'points at infinity'.

        Returns
        -------
        regions : list of tuples
            Indices of vertices in each revised Voronoi regions.
        vertices : list of tuples
            Coordinates for revised Voronoi vertices. Same as coordinates
            of input vertices, with 'points at infinity' appended to the
            end.

        """

        if vor.points.shape[1] != 2:
            raise ValueError("Requires 2D input")

        new_regions = []
        new_vertices = vor.vertices.tolist()

        center = vor.points.mean(axis=0)
        if radius is None:
            radius = vor.points.ptp().max()

        # Construct a map containing all ridges for a given point
        all_ridges = {}
        for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
            all_ridges.setdefault(p1, []).append((p2, v1, v2))
            all_ridges.setdefault(p2, []).append((p1, v1, v2))

        # Reconstruct infinite regions
        for p1, region in enumerate(vor.point_region):
            vertices = vor.regions[region]

            if all(v >= 0 for v in vertices):
                # finite region
                new_regions.append(vertices)
                continue

            # reconstruct a non-finite region
            ridges = all_ridges[p1]
            new_region = [v for v in vertices if v >= 0]

            for p2, v1, v2 in ridges:
                if v2 < 0:
                    v1, v2 = v2, v1
                if v1 >= 0:
                    # finite ridge: already in the region
                    continue

                # Compute the missing endpoint of an infinite ridge

                t = vor.points[p2] - vor.points[p1]  # tangent
                t /= np.linalg.norm(t)
                n = np.array([-t[1], t[0]])  # normal

                midpoint = vor.points[[p1, p2]].mean(axis=0)
                direction = np.sign(np.dot(midpoint - center, n)) * n
                far_point = vor.vertices[v2] + direction * radius

                new_region.append(len(new_vertices))
                new_vertices.append(far_point.tolist())

            # sort region counterclockwise
            vs = np.asarray([new_vertices[v] for v in new_region])
            c = vs.mean(axis=0)
            angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
            new_region = np.array(new_region)[np.argsort(angles)]

            # finish
            new_regions.append(new_region.tolist())

        return new_regions, np.asarray(new_vertices)

    vor = Voronoi(extracted_particles)

    # plot
    regions, vertices = voronoi_finite_polygons_2d(vor)

    # colorize
    if colors is None:
        for region in regions:
            polygon = vertices[region]
            plt.fill(*zip(*polygon), alpha=0.3)
    else:
        for idx, region in enumerate(regions):
            polygon = vertices[region]
            plt.fill(*zip(*polygon), alpha=0.3, color=colors[idx])

    plt.plot(extracted_particles[:, 0], extracted_particles[:, 1], 'ko')
    if xlim is None:
        xlim = (vor.min_bound[0] - 0.5, vor.max_bound[0] + 0.5)
    if ylim is None:
        ylim = (vor.min_bound[1] - 0.5, vor.max_bound[1] + 0.5)
    plt.xlim(xlim[0], xlim[1])
    plt.ylim(ylim[0], ylim[1])



