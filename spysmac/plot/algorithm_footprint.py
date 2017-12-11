import os
import logging
import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy import spatial

from smac.configspace import Configuration
from smac.runhistory.runhistory import RunHistory

from spysmac.utils.helpers import get_cost_dict_for_config, get_timeout

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
                 output_dir=""):
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
        self.rng = np.random.RandomState()  # TODO random over module...

        self.rh = rh
        self.insts = list(inst_feat.keys())  # This is the order of instances!
        if self.output_dir and not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.algorithms = algorithms.keys()  # Configs
        self.algo_names = algorithms         # Maps config -> name
        self.algo_performance = {}           # Maps instance -> performance
        self.algo_labels = {}                # Maps config -> label

        self.features = np.array([inst_feat[k] for k in self.insts])
        self.features_2d = self.reduce_dim(self.features)
        self.clusters, self.cluster_dict = self.get_clusters(self.features_2d)

        self.cutoff = cutoff

        self.label_instances()

    def get_performance(self, algorithm, instance):
        """
        Return performance according to (possibly EPM-)validated runhistory.
        """
        if not algorithm in self.algo_performance:
            self.algo_performance[algorithm] = get_cost_dict_for_config(self.rh, algorithm)
        return self.algo_performance[algorithm][instance]

    def footprint(self, a, density_threshold, purity_threshold):
        """ Footprint for algorithm a with density and purity thresholds
        (algorithm 1 in Smith-Miles 2014) """
        good = [i for i in self.insts if self.algo_labels[a][i] == 1]
        if len(good) < 3:
            return
        bad = [i for i in self.insts if not i in good]

        def get_2NN(pos, insts):
            """ compare all insts to pos. """
            # (position, distance)
            nearest, sec_nearest = (None, np.inf), (None, np.inf)
            for tmp_pos in insts:
                if np.all(pos == tmp_pos):
                    continue
                dist = np.linalg.norm(tmp_pos - pos)
                if dist < nearest[1]:
                    sec_nearest = nearest
                    nearest = (tmp_pos, dist)
                elif dist < sec_nearest[1]:
                    sec_nearest = (tmp_pos, dist)
            return (nearest[0], sec_nearest[0])

        regions = {}
        # Only single points in not_in_region - as tuples, hashable.
        not_in_region = set([tuple(f) for f in self.features_2d])

        # While there are still at least two points not part of any region
        while (len(not_in_region) >= 3):
            # Select random good instance TODO also from in_regions?!?!
            rand_good = self.rng.choice(good)
            rand_pos = self.features_2d[self.insts.index(rand_good)]
            # Find two closest neighbors not in region
            one, two = get_2NN(rand_pos, not_in_region)
            triangle = np.array((rand_pos, one, two))
            #self.logger.debug(triangle)
            centroid = np.sum(triangle, axis=0)/len(triangle)
            regions[tuple(centroid)] = (one, two, rand_pos)
            for p in triangle:
                try: not_in_region.remove(tuple(p))
                except KeyError: pass
            #self.logger.debug("Not in region: %d", len(not_in_region))

        stop = False
        while not stop:
            stop = True
            for cent in regions.keys():
                #self.logger.debug(regions)
                reg = regions[cent]
                cent_array = np.array(cent)
                nearest_cent = get_2NN(cent_array, [np.array(r) for r in
                                        regions.keys()])[0]
                nearest_reg = regions[tuple(nearest_cent)]
                # Check purity and density
                new_reg = np.vstack((reg, nearest_reg))
                combined_hull = spatial.ConvexHull(new_reg)
                density = len(new_reg)/combined_hull.area
                purity = (len([i for i in reg if i in good]) + len([i for i
                           in nearest_reg if i in good])) / len(new_reg)
                if density > density_threshold and purity > purity_threshold:
                    regions.pop(tuple(cent))
                    regions.pop(tuple(nearest_cent))
                    new_centroid = tuple(np.sum(new_reg, axis=0)/len(new_reg))
                    regions[centroid] = new_reg
                    stop = False
                    break

        # We now have final regions -> return sum of individual convex hulls
        area = 0
        self.logger.debug(regions)
        self.logger.debug(len(regions))
        for r in regions.values():
            self.logger.debug(r)
            hull = spatial.ConvexHull(r)
            area += hull.area
        self.logger.debug("Area for %s is %f", self.algo_names[a], area)
        return area

    def label_instances(self, epsilon=0.95):
        """
        Returns dictionary with a label for each instance.

        Returns
        -------
        labels: Dict[str:float]
            maps instance-names (strings) to label (floats)
        """
        start = time.time()
        self.algo_labels = {a:{} for a in self.algorithms}
        for i in self.insts:
            performances = [self.get_performance(a, i) for a in self.algorithms]
            best_performance = min(performances)
            for a in self.algorithms:
                performance = self.get_performance(a, i)
                #self.logger.debug("%s on \'%s\': best/this (%f/%f=%f)",
                #                  self.algo_names[a], i,
                #                  best_performance, performance,
                #                  best_performance / performance)
                if (performance == 0 or
                    (best_performance/performance >= epsilon and
                     not performance >= self.cutoff)):
                    # Algorithm for instance is in threshhold epsilon
                    #   and no timeout
                    label = 1
                else:
                    label = 0
                self.algo_labels[a][i] = label
        self.logger.debug("Labeling instances in %.2f secs.", time.time() - start)

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
        outpaths = []

        for e in np.hstack([np.arange(0.0, 1.0, 0.05), np.arange(0.96, 1.0, 0.005)]):
            self.label_instances(e)
            for a in self.algorithms:
                # Plot without clustering (for all insts)
                suffix = 'all_{:4.3f}.png'.format(e)
                path = os.path.join(self.output_dir,
                                    '_'.join(
                                    [self.algo_names[a], suffix]))
                self.logger.debug("Plot saved to '%s'", path)
                outpaths.append(self._plot_points(a, path))
                # Plot per cluster
                for c in self.cluster_dict.keys():
                    path = os.path.join(self.output_dir,
                                        '_'.join([self.algo_names[a], str(c)+'.png']))
                    outpaths.append(self._plot_points(a, path, self.cluster_dict[c]))
        return outpaths

    def _plot_points(self, conf, out, insts=[]):
        """ Plot good versus bad for conf. Mainly for debugging labels!

        Parameters
        ----------
        conf: Configuration
            configuration for which to plot good vs bad
        out: str
            output path
        insts: List[str]
            instances to be plotted

        Returns
        -------
        outpath: str
            output path
        """
        fig, ax = plt.subplots()

        if len(insts) == 0:
            insts = self.insts

        good, bad = [], []
        for k, v in self.algo_performance[conf].items():
            # Only consider passed insts
            if not k in insts:
                continue
            # Append insts to plot either to good or bad
            point = self.features_2d[self.insts.index(k)]
            if self.algo_labels[conf][k] == 0:
                bad.append(point)
            else:
                good.append(point)
        good, bad = np.array(good), np.array(bad)

        if len(bad) > 0: ax.scatter(bad[:, 0], bad[:, 1], color="red", s=6)
        if len(good) > 0: ax.scatter(good[:, 0], good[:, 1], color="green", s=6,
                alpha=0.8)
        fig.suptitle(self.algo_names[conf])
        ax.set_ylabel('Principal Component 1')
        ax.set_xlabel('Principal Component 2')
        fig.savefig(out)
        plt.close(fig)

        return out

    def reduce_dim(self, feature_array):
        """ Expects feature-array (not dict!)

        Parameters
        ----------
        feature_array: np.array
            array containing features in order of self.inst_names

        Returns
        -------
        feature_array_2d: np.array
            array with pca'ed features (2-dimensional)
        """
        # Perform PCA to reduce features to 2
        n_feats = feature_array.shape[1]
        if n_feats > 2:
            self.logger.debug("Use PCA to reduce features to two dimensions")
            ss = StandardScaler()
            feature_array = ss.fit_transform(feature_array)
            pca = PCA(n_components=2)
            feature_array = pca.fit_transform(feature_array)
        return feature_array

    def get_clusters(self, features_2d):
        """ Mapping instances to clusters, using silhouette-scores to determine
        number of cluster.

        Parameters
        ----------
        features_2d: np.array
            scaled and pca'ed feature-array (2D)

        Returns
        -------
        clusters: np.array
            in the order of self.insts, clusters of instances
        cluster_dict: Dict[int: int]
            maps clusters to indices of instances, i.e. for instances range(10):
            {0: [1,3,5], 1: [2,7,8], 2: [0,4,6,9]}
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
