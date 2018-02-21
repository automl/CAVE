import os
import logging
import time
from collections import OrderedDict
import itertools

import numpy as np
import matplotlib.pyplot as plt
plt.style.use(os.path.join(os.path.dirname(__file__), 'mpl_style'))
import matplotlib.lines as mlines
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy import spatial
import pandas as pd

from smac.configspace import Configuration
from smac.runhistory.runhistory import RunHistory

from cave.utils.helpers import get_cost_dict_for_config, get_timeout

__author__ = "Joshua Marben"
__copyright__ = "Copyright 2017, ML4AAD"
__license__ = "3-clause BSD"
__maintainer__ = "Joshua Marben"
__email__ = "joshua.marben@neptun.uni-freiburg.de"

class AlgorithmFootprint(object):
    """ Class that provides the algorithmic footprints after
     "Measuring algorithm footprints in instance space"
     (Kate Smith-Miles, Kate Smith-Miles)
     ...
     TODO

     General procedure:
         - label for each algorithm each instance with the same metric
         - map the instances onto a plane using pca
    """
    def __init__(self, rh: RunHistory, inst_feat, algorithms, cutoff=np.inf,
                 output_dir="", rng=None):
        """
        Parameters
        ----------
        rh: RunHistory
            runhistory to take performance from
        inst_feat: dict[str->np.array]
            instances names mapped to features
        algorithms: Dict[Configuration->str]
            mapping configs to names (here just: default, incumbent)
        cutoff: int
            cutoff (if available)
        output_dir: str
            output directory
        """
        self.logger = logging.getLogger(
            self.__module__ + '.' + self.__class__.__name__)
        self.output_dir = output_dir
        self.rng = rng
        if not self.rng:
            self.logger.info("No randomstate passed. Generate deterministic "
                             "random state.")
            self.rng = np.random.RandomState(42)

        self.rh = rh
        self.insts = list(inst_feat.keys())  # This is the order of instances!
        if self.output_dir and not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.algorithms = algorithms.keys()  # Configs
        self.algo_names = algorithms         # Maps config -> name
        self.algo_performance = {}           # Maps instance -> performance
        self.algo_labels = {}                # Maps config -> label

        self.features = np.array([inst_feat[k] for k in self.insts])
        self.features_2d = self._reduce_dim(self.features, 2)
        self.features_3d = self._reduce_dim(self.features, 3)
        self.clusters, self.cluster_dict = self.get_clusters(self.features_2d)

        self.cutoff = cutoff

        self._label_instances()

    def _reduce_dim(self, feature_array, n=2):
        """ Expects feature-array (not dict!)

        Parameters
        ----------
        feature_array: np.array
            array containing features in order of self.inst_names
        n: int
            target dimension for pca, 2 or 3

        Returns
        -------
        feature_array_nd: np.array
            array with pca'ed features (n-dimensional)
        """
        if n not in [2, 3]:
            raise ValueError("Only 2 and 3 supported as target dimension!")
        # Perform PCA to reduce features to n
        n_feats = feature_array.shape[1]
        if n_feats > n:
            self.logger.debug("Use PCA to reduce features to %d dimensions", n)
            ss = StandardScaler()
            feature_array = ss.fit_transform(feature_array)
            pca = PCA(n_components=n)
            feature_array = pca.fit_transform(feature_array)
        return feature_array

    def _get_performance(self, algorithm, instance):
        """
        Return performance according to (possibly EPM-)validated runhistory.
        """
        if not algorithm in self.algo_performance:
            self.algo_performance[algorithm] = get_cost_dict_for_config(self.rh, algorithm)
        return self.algo_performance[algorithm][instance]

    def _label_instances(self, epsilon=0.95):
        """
        Returns dictionary with a label for each instance.

        Returns
        -------
        labels: Dict[str:float]
            maps instance-names (strings) to label (floats)
        """
        start = time.time()
        if len(self.algo_labels) > 0:
            return
        self.algo_labels = {a:{} for a in self.algorithms}
        for i in self.insts:
            performances = [self._get_performance(a, i) for a in self.algorithms]
            best_performance = min(performances)
            for a in self.algorithms:
                performance = self._get_performance(a, i)
                #self.logger.debug("%s on \'%s\': best/this (%f/%f=%f)",
                #                  self.algo_names[a], i,
                #                  best_performance, performance,
                #                  best_performance / performance)
                if (performance == 0 or
                    (best_performance/performance >= epsilon and
                     not performance >= self.cutoff)):
                    # Algorithm for instance is in threshhold epsilon and no timeout
                    label = 1
                else:
                    label = 0
                self.algo_labels[a][i] = label
        self.logger.debug("Labeling instances in %.2f secs.", time.time() - start)

####### FOOTPRINT

    def footprint(self, a, density_threshold, purity_threshold):
        """
        Calculating the footprint within a portfolio using convex hulls that
        depend on density and purity thresholds.
        (algorithm 1 in Smith-Miles 2014)

        We use 3 ways to refer to an instance here:
        name: the name (unique!) of the instance
        feat2d: the position as np.array
        tup: the tuple-version of feat2d (hashable...)

        Parameters
        ----------
        a: Configuration
            configuration to get footprint of
        density_threshold: float
            minimum density that regions must show to be merged
        purity_threshold: float
            minimum purity (percentage of good instance)
            that regions must show to be merged

        Returns
        -------
        footprint: float
            the size of all resulting convex hulls
        """
        def get_2NN(x, X):
            """ Return indices in X of two nearest points in X to x. """
            # map index to dist from point
            dist = [(i, np.linalg.norm(tmp - x)) for i, tmp in enumerate(X)]
            # sort after dist
            dist = sorted(dist, key=lambda x: x[1])
            # return indices of the two smallest values
            return (dist[0][0], dist[1][0])

        count_exceptions = 0

        ### Initialise Stage
        # Map inst-names to feat2d (np.array) and tup (tuple)
        inst_feat2d = {i:self.features_2d[idx] for idx, i in enumerate(self.insts)}
        inst_tup = {i:tuple(pos) for i, pos in inst_feat2d.items()}

        # regions maps tuple(centroid) of region to inst-names in region
        regions = OrderedDict()

        # Instances (by name) in not_in_region
        not_in_region = self.insts[:]

        # Randomly select a good instance;
        good = [i for i in self.insts if self.algo_labels[a][i] == 1]
        if len(good) < 3:
            self.logger.debug("Less than 3 good instances found in %s, footprint"
                              " not calculated.", self.algo_names[a])
            return 0

        # Repeat until no more triangles can be formed (at least 3 points left).
        while (len(not_in_region) >= 3):
            # Select random good instance TODO also from in_regions?!?!
            rand_good = self.rng.choice(good)
            try: not_in_region.remove(rand_good)  # Remove here so it's not its own nearest neighbor
            except ValueError: pass

            # Form a closed region (triangle) with the two closest (smallest
            #        Euclidean distance in feature space) instances to
            #        rand_good, not already part of a triangle;
            idx1, idx2 = get_2NN(inst_feat2d[rand_good],
                                 [inst_feat2d[i] for i in not_in_region])

            triangle = (rand_good, not_in_region[idx1], not_in_region[idx2])  # names
            triangle_feat = np.array([inst_feat2d[i] for i in triangle])
            centroid = np.sum(np.array(triangle_feat), axis=0)/len(triangle)
            regions[tuple(centroid)] = triangle
            for p in triangle:
                try: not_in_region.remove(p)
                except ValueError: pass

        ### Merge Stage
        # Repeat the Merge Stage until there are no more pairs to consider.
        # If we iterated over whole list once, we are done.
        stop = False
        while not stop:
            stop = True
            centroids = list(regions.keys())
            self.rng.shuffle(centroids)
            # Randomly select a closed region;
            for idx, cent in enumerate(centroids):
                reg = regions[cent]          # inst-names!
                cent_array = np.array(cent)  # Keys in dict are tuples!

                # Find the closest closed region (minimum Euclidean
                #   centroid distance);
                remaining_centroids = [np.array(c) for c in regions.keys() if
                                        not c == cent]
                idx = get_2NN(cent_array, remaining_centroids)[0]
                nearest_cent = tuple(remaining_centroids[idx])
                nearest_reg = regions[nearest_cent]  # inst-names!

                # Check purity and density
                new_reg = tuple(set(reg) | set(nearest_reg))  # names
                new_reg_array = np.array([inst_feat2d[i] for i in new_reg]) # array
                try:
                    combined_hull = spatial.ConvexHull(new_reg_array)
                except spatial.qhull.QhullError:
                    count_exceptions += 1
                    continue
                density = len(new_reg)/combined_hull.volume
                purity = (len([i for i in reg if i in good]) +
                          len([i for i in nearest_reg if i in good])) / float(len(new_reg))
                if density > density_threshold and purity > purity_threshold:
                    self.logger.debug("Purity: %f, density: %f", purity, density)
                    regions.pop(cent)
                    regions.pop(nearest_cent)
                    new_centroid = tuple(np.sum(new_reg_array, axis=0)/len(new_reg))
                    regions[new_centroid] = new_reg
                    stop = False
                    break

        # We now have final regions -> return sum of individual convex hulls
        area = 0
        for reg in regions.values():
            try:
                hull = spatial.ConvexHull(np.array([inst_feat2d[p] for p in reg]))
                area += hull.volume
            except spatial.qhull.QhullError:
                count_exceptions += 1
                pass
        self.logger.debug("Area for %s is %f (%d Qhull-exceptions, %d/%d good "
                          "insts, %d regions)",
                          self.algo_names[a], area, count_exceptions, len(good),
                          len(self.insts), len(regions))
        return area

####### PLOTS

    def _get_rgba(self, all, good, bad):
        """ Calculates the red and green parts of the individual dots.
        The red part is the number of points on the same coordinate belonging to the bad group
        divided by the number of all points on the same coordinate, same for
        green part.

        Parameters:
        -----------
        all: list
            list of features of all instances
        good: list
            instances labeled good
        bad: list
            instances labeled bad

        Returns:
        --------
        r_g_b_a: list
            list of tuples with rgba-values
        zorder: list
            zorder values
        """
        len_longest = min(len(good), len(bad))
        colors, alpha, zorder = [], [], []
        counts = all.groupby(all.columns.tolist(), as_index=False).size()  # count the occurance of values
        if len(good) > 0: counts_g = good.groupby(good.columns.tolist(),
                as_index=False).size()#.unstack()  # in good
        if len(bad) > 0: counts_b = bad.groupby(bad.columns.tolist(),
                as_index=False).size()#.unstack()  # and bad
        for idx, coords in enumerate(all.values):  # individually plot the points
            self.logger.debug(counts)
            r, g, b = 0, 0, 0
            if len(bad) > 0 and len(coords) == 3:
                try:
                    r = counts_b[coords[0]][coords[1]][coords[2]] / counts[coords[0]][coords[1]][coords[2]]
                except KeyError:
                    pass
            if len(good) > 0 and len(coords) == 3:
                try:
                    g = counts_g[coords[0]][coords[1]][coords[2]] / counts[coords[0]][coords[1]][coords[2]]
                except KeyError:
                    pass
            if len_longest < idx:  # if we plot points from the shorter list, increase zorder and use small alpha
                alpha = 0.375
                zorder.append(9999)
            else:
                zorder.append(1)
                alpha = 1
            colors.append((r, g, b, alpha))
        return np.array(colors), zorder

    def _get_good_bad(self, conf, insts=[]):
        """ Creates a list of indices for good and bad instances for a
        configuration.

        Parameters
        ----------
        conf: Configuration
            configuration for which to plot good vs bad
        insts: List[str]
            instances to be plotted

        Returns
        -------
        outpath: str
            output path
        """
        if len(insts) == 0:
            insts = self.insts

        good_idx, bad_idx = [], []
        for k, v in self.algo_performance[conf].items():
            # Only consider passed insts
            if not k in insts:
                continue
            # Append inst-idx either to good or to bad
            if self.algo_labels[conf][k] == 0:
                bad_idx.append(self.insts.index(k))
            else:
                good_idx.append(self.insts.index(k))
        assert(len(good_idx) == len(set(good_idx)))
        assert(len(bad_idx) == len(set(bad_idx)))
        good_idx, bad_idx = np.array(good_idx), np.array(bad_idx)
        self.logger.debug("for config %s good: %d, bad: %d",
                          self.algo_names[conf], len(good_idx), len(bad_idx))
        return (good_idx, bad_idx)

    def plot2d(self):
        """ Plot shaded 2d-version of the algorithm footprint. """
        plots = []
        for a in self.algorithms:
            # Plot without clustering (for all insts)
            out_fn = os.path.join(self.output_dir, 'footprint_' +
                                  self.algo_names[a] + '_2d.png')
            self.logger.debug("Plot saved to '%s'", out_fn)
            fig, ax = plt.subplots()
            good_idx, bad_idx = self._get_good_bad(a)
            # As we don't have such a high resolution when plotting, i.e. we don't see differences between 0.001 and 0.00001
            # all points that lie close by might overlap completely. To easily spot these, squash everything down to one
            # decimal
            good = np.array([self.features_2d[idx] for idx in good_idx])
            bad = np.array([self.features_2d[idx] for idx in bad_idx])
            good, bad = np.around(np.array(good), decimals=1), np.around(np.array(bad), decimals=1)

            # working with dataframes to get easy counts to use for plotting
            good = pd.DataFrame(good)
            bad = pd.DataFrame(bad)
            if len(good) < len(bad):  # decide which to plot first. (short list unlikely to shadow many points in long list)
                all = pd.concat([bad, good])
            else:
                all = pd.concat([good, bad])
            len_longest = min(len(good), len(bad))
            counts = all.groupby(all.columns.tolist(), as_index=False).size()  # count the occurance of values
            if len(good) > 0: counts_g = good.groupby(good.columns.tolist(), as_index=False).size().unstack()  # in good
            if len(bad) > 0: counts_b = bad.groupby(bad.columns.tolist(), as_index=False).size().unstack()  # and bad
            for idx, coords in enumerate(all.values):  # individually plot the points
                r, g, b = 0, 0, 0
                if len(bad) > 0 and coords[0] in counts_b.index and coords[1] in counts_b.columns:  # determine red part
                    # red part is the number of points on the same coordinate belonging to the bad group
                    # divided by the number of all points on the same coordinate
                    r = counts_b[coords[1]][coords[0]] / counts[coords[0]][coords[1]]
                if len(good) > 0 and coords[0] in counts_g.index and coords[1] in counts_g.columns: # similar for green
                    g = counts_g[coords[1]][coords[0]] / counts[coords[0]][coords[1]]
                zorder = 1
                alpha = 1
                if len_longest < idx:  # if we plot points from the shorter list, increase zorder and use small alpha
                    alpha = 0.375
                    zorder=9999
                plt.scatter(coords[0], coords[1], color=(r, g, b), s=15, zorder=zorder, alpha=alpha)
            ax.set_ylabel('principal component 1')
            ax.set_xlabel('principal component 2')
            plt.tight_layout()
            fig.savefig(out_fn)
            plt.close(fig)
            plots.append(out_fn)
        return plots

    def plot3d(self):
        """ Plot 3d-version of the algorithm footprint from four different
        angles. """
        plots = []
        for a in self.algorithms:
            # Plot without clustering (for all insts)
            out_fns = [os.path.join(self.output_dir, 'footprint_' +
                      self.algo_names[a] + '_3d_{}.png'.format(i)) for i in range(4)]
            self.logger.debug("Plot saved to '%s'", out_fns)
            fig, ax = plt.subplots()
            good_idx, bad_idx = self._get_good_bad(a)
            good = np.array([self.features_3d[idx] for idx in good_idx])
            bad = np.array([self.features_3d[idx] for idx in bad_idx])
            axes = {0 : 'principal component 1',
                    1 : 'principal component 2',
                    2 : 'principal component 3'}
            for out_fn, axes_ordered in zip(out_fns,
                    list(itertools.permutations([0, 1, 2]))[:len(out_fns)]):
                # Plot 3d
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                x, y, z = axes_ordered
                if len(good) > 0: ax.scatter(xs=good[:, x], ys=good[:, y],
                                             zs=good[:, z], color="green")
                if len(bad) > 0: ax.scatter(xs=bad[:, x], ys=bad[:, y],
                                            zs=bad[:, z], color="red")
                ax.set_xlabel(axes[x], fontsize=12)
                ax.set_ylabel(axes[y], fontsize=12)
                ax.set_zlabel(axes[z], fontsize=12)
                plt.tight_layout()
                #for out_fn, angle in zip(out_fns, range(20, 381, 90)):
                #    ax.view_init(30, angle)
                fig.savefig(out_fn)
                plt.close(fig)
            plots.append(out_fns)
        return plots

####### CLUSTER

    def plot_points_per_cluster(self):
        """ Plot good versus bad for passed config per cluster.

        Parameters
        ----------
        conf: Configuration
            configuration for which to plot good vs bad
        out: str
            output path

        Returns
        -------
        outpaths: List[str]
            output paths per cluster
        """
        # For Development/Debug:
        algo_fp_debug = os.path.join(self.output_dir, 'debug', 'algo_fp')
        if not os.path.exists(algo_fp_debug):
            os.makedirs(algo_fp_debug)
        for e in np.hstack([np.arange(0.0, 1.0, .95), np.arange(0.96, 1.0, 0.02)]):
            self._label_instances(e)
            for a in self.algorithms:
                # Plot without clustering (for all insts)
                suffix = 'all_{:4.3f}.png'.format(e)
                path = os.path.join(algo_fp_debug,
                                    '_'.join([self.algo_names[a], suffix]))
                path = self.plot2d(a, path)
                self.logger.debug("Plot saved to '%s'", path)
        self._label_instances()
        for c in self.cluster_dict.keys():
            # Plot per cluster
            path = os.path.join(algo_fp_debug, 'cluster_' + str(c) + '_fp_' +
                                               self.algo_names[a] + '_0.95.png')
            path = self.plot2d(a, path, self.cluster_dict[c])

    def get_clusters(self, features_2d):
        """ Mapping instances to clusters, using silhouette-scores to determine
        number of cluster.

        Returns
        -------
        paths: List[str]
            paths to plots
        """
        # get silhouette scores for k_means with 2 to 12 clusters
        # use number of clusters with highest silhouette score
        best_score, best_n_clusters = -1, -1
        min_clusters, max_clusters = 2, 12
        clusters = None
        for n_clusters in range(min_clusters, max_clusters):
            km = KMeans(n_clusters=n_clusters)
            y_pred = km.fit_predict(features_2d)
            score = silhouette_score(features_2d, y_pred)
            if score > best_score:
                best_n_clusters = n_clusters
                best_score = score
                clusters = y_pred

        self.logger.debug("%d clusters detected using silhouette scores",
                          best_n_clusters)

        cluster_dict = {n:[] for n in range(best_n_clusters)}
        for i, c in enumerate(clusters):
            cluster_dict[c].append(self.insts[i])

        self.logger.debug("Distribution over clusters: %s",
                          str({k:len(v) for k, v in cluster_dict.items()}))

        return clusters, cluster_dict
