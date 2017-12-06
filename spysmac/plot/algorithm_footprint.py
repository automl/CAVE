import os
import logging
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
        self.rh = rh
        self.insts = list(inst_feat.keys())  # This is the order of instances!
        self.output_dir = output_dir

        self.algorithms = algorithms.keys()  # Configs
        self.algo_names = algorithms  # Maps config -> name
        self.algo_performance = {}  # Maps instance -> performance

        self.features = np.array([inst_feat[k] for k in self.insts])
        self.features_2d = self.reduce_dim(self.features)
        self.clusters, self.cluster_dict = self.get_clusters(self.features_2d)

        self.cutoff = cutoff

        self.label_instances()

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

        self.logger.debug("Distribution over clusters: %s", str(cluster_dict))

        raise Exception()
        return clusters, cluster_dict

    def get_performance(self, algorithm, instance):
        """
        Return performance according to runhistory (or EPM???)
        """
        if not algorithm in self.algo_performance:
            self.algo_performance[algorithm] = get_cost_dict_for_config(self.rh, algorithm)
        return self.algo_performance[algorithm][instance]

    def label_instances(self, epsilon=0.95):
        """
        Returns dictionary with a label for each instance.

        Returns
        -------
        labels: Dict[str:float]
            maps instance-names (strings) to label (floats)
        """
        self.algo_labels = {a:{} for a in self.algorithms}
        for i in self.insts:
            performances = [self.get_performance(a, i) for a in self.algorithms]
            self.logger.debug(performances)
            best_performance = min(performances)
            for a in self.algorithms:
                self.logger.debug("%s on \'%s\': best/this (%f/%f=%f)",
                                  self.algo_names[a], i,
                                  best_performance, self.get_performance(a, i),
                                  best_performance/self.get_performance(a, i))
                if (self.get_performance(a, i) == 0 or
                    (best_performance/self.get_performance(a, i)) > epsilon):
                    # Algorithm for instance is in threshhold epsilon
                    label = 1
                else:
                    label = 0
                self.logger.debug(label)
                self.algo_labels[a][i] = label

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

        for a in self.algorithms:
            # Plot without clustering (for all insts)
            path = os.path.join(self.output_dir,
                                '_'.join([self.algo_names[a], 'all.png']))
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

        if len(bad) > 0: ax.scatter(bad[:, 0], bad[:, 1], color="red", s=3)
        if len(good) > 0: ax.scatter(good[:, 0], good[:, 1], color="green", s=3)
        fig.suptitle(self.algo_names[conf])
        ax.set_ylabel('Principal Component 1')
        ax.set_xlabel('Principal Component 2')
        fig.savefig(out)
        plt.close(fig)

        return out

#### Below not implemented """

    def get_footprint(self, default, incumbent):
        """ Calculate footprint by comparing overall convex hull to hulls of def
        and inc. Also comparing the intersection of hulls of def and inc.

        conf: Configuration
            configuration for which to return hull
        """
        raise NotImplemented()
        # get labels for both configs
        def_labels = self.label_instances(default)
        def_label_list = [def_labels[i] for i in self.inst_names]
        inc_labels = self.label_instances(incumbent)
        inc_label_list = [inc_labels[i] for i in self.inst_names]
        # TODO: labeling strategy?
        def_good = [k for k, v in def_labels.items() if v == 1]
        inc_good = [k for k, v in inc_labels.items() if v == 1]
        def_bad = [k for k, v in def_labels.items() if v == 0]
        inc_bad = [k for k, v in inc_labels.items() if v == 0]
        # get hulls
        hulls = self.get_convex_hulls(self.cluster_dict)
        def_hulls = self.get_convex_hulls(self.cluster_dict, def_good)
        inc_hulls = self.get_convex_hulls(self.cluster_dict, inc_good)

        # get footprints
        def_footprint = self.hulls_area(def_hulls)/self.hulls_area(hulls)
        inc_footprint = self.hulls_area(inc_hulls)/self.hulls_area(hulls)
        self.logger.info("Footprints: Default=%f, Incumbent=%f", def_footprint,
                                                                 inc_footprint)
        # TODO intersection
        for c in range(len(self.cluster_dict)):
            self.plot_hulls(def_hulls, inc_hulls)

        return def_footprint, inc_footprint

    def get_convex_hulls(self, cluster_dict, insts=None):
        """ Get convex hulls per cluster. 

        Parameters
        ----------
        cluster_dict: Dict[int: str]
            maps clusters to instances, i.e. for instances range(10):
            {0: ["1","3","5"], 1: ["2","7","8"], 2: ["0","4","6","9"]}
        insts: List[str]
            instances to consider (defined through labeling for each config)

        Returns
        -------
        hulls: List[ConvexHull]
            list with convex hulls for each cluster. clusters in increasing
            order, i.e. [hull_cluster_0, hull_cluster_1, hull_cluster_2, ...]
        """
        raise NotImplemented()
        self.logger.debug("Calculating convex hulls for algorithm footprint.")
        if not insts: insts = self.insts
        hulls = []
        # For each cluster get convex hull
        for cluster in range(len(cluster_dict)):
            indexes_in_cluster = cluster_dict[cluster]
            instances = [inst for inst in self.insts
                         if inst in indexes_in_cluster and inst in insts]
            self.logger.debug("Hull for cluster %d with %d instances", cluster,
                    len(instances))
            if len(instances) < 3:
                hulls.append(0)
                continue
            inst_indexes = [self.inst_names.index(i) for i in instances]
            points = [self.features_2d[i] for i in inst_indexes]
            hull = spatial.ConvexHull(points, qhull_options="Qt")
            hulls.append(hull)
        return hulls

    def plot_hulls(self, hulls1, hulls2):
        """ Create a plot for each cluster, plotting hulls1 vs hulls2.

        Returns
        -------
        paths: List[str]
            paths to plots
        """
        raise NotImplemented()
        paths = []
        for i, h in enumerate(hulls1):
            output = os.path.join(self.output, "algorithm_footprint_hulls_cluster_"+str(i)+".png")
            self.logger.debug("plotting cluster %d to \"%s\"", i, output)
            # Get coordinates for insts in cluster
            inst_indexes = [self.inst_names.index(j) for j in self.inst_names
                            if j in self.cluster_dict[i]]
            points = np.array([self.features_2d[j] for j in inst_indexes])
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(points[:,0], points[:,1], 'o')
            h1, h2 = hulls1[i], hulls2[i]
            if not h1 == 0:
                for simplex in h1.simplices:
                    ax.plot(points[simplex, 0], points[simplex, 1], 'r-.')
            if not h2 == 0:
                for simplex in h2.simplices:
                    ax.plot(points[simplex, 0], points[simplex, 1], 'b--')
            # Add legend
            red_line = mlines.Line2D([], [], color='red', ls='-.',
                                              markersize=15, label='default')
            blue_line = mlines.Line2D([], [], color='blue', ls='--',
                                              markersize=15, label='incumbent')
            ax.legend(handles=[blue_line, red_line])

            paths += output
            fig.savefig(output)
            plt.close(fig)
        return paths

    def hulls_area(self, hulls):
        """ Sum up area of hulls over clusters.

        Parameters
        ----------
        hulls: List[ConvexHull]
            convex hulls

        Returns
        -------
        area: float
            total area of all hulls
        """
        raise NotImplemented()
        area = 0.0
        for h in hulls:
            if not h:
                continue
            area += h.area
        return area

