import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy import spatial

from smac.configspace import Configuration
from smac.runhistory.runhistory import RunHistory

from spysmac.utils.helpers import get_cost_dict_for_config, get_timeout


class AlgorithmFootprint(object):
    """ Class that provides the algorithmic footprints after
     "Measuring algorithm footprints in instance space"
     (Kate Smith-Miles, Kate Smith-Miles)
     ...

     General procedure:
         - label for each algorithm each instance with the same metric
         - map the instances onto a plane using pca
         - calculate the hulls of the instances
         - compare the hulls to the hull of all instances
    """
    def __init__(self, rh:RunHistory, inst_feat, cutoff, output, plotter=None):
        """
        Parameters
        ----------
        inst_feat: dict[feature-vectors]
            instances names mapped to features
        """
        self.logger = logging.getLogger(
            self.__module__ + '.' + self.__class__.__name__)
        self.rh = rh
        self.inst_names = list(inst_feat.keys())  # This is the order of instances!
        self.output = output

        self.features = np.array([inst_feat[k] for k in self.inst_names])
        self.features_2d = self.reduce_dim(self.features)
        self.clusters, self.cluster_dict = self.get_clusters(self.features_2d)

        self.plotter = plotter
        self.cutoff = cutoff

    def get_footprint(self, default, incumbent):
        """ Calculate footprint by comparing overall convex hull to hulls of def
        and inc. Also comparing the intersection of hulls of def and inc.

        conf: Configuration
            configuration for which to return hull
        """
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

    def plot_hulls(self, hulls1, hulls2):
        """ Create a plot for each cluster, plotting hulls1 vs hulls2.

        Returns
        -------
        paths: List[str]
            paths to plots
        """
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
                ax.plot(points[h1.vertices,0], points[h1.vertices,1],
                        'r--', lw=2)
                ax.plot(points[h1.vertices[0],0], points[h1.vertices[0],1],
                        'r--')
                #for simplex in h1.simplices:
                #    ax.plot(points[simplex, 0], points[simplex, 1], 'r-')
            if not h2 == 0:
                ax.plot(points[h2.vertices,0], points[h2.vertices,1],
                        'b--', lw=2.5)
                ax.plot(points[h2.vertices[0],0], points[h2.vertices[0],1],
                        'b--')
                #for simplex in h2.simplices:
                #    ax.plot(points[simplex, 0], points[simplex, 1], 'b-')
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
        area = 0.0
        for h in hulls:
            if not h:
                continue
            area += h.area
        return area

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

    def get_clusters(self, features):
        """ Mapping instances to clusters, using silhouette-scores to determine
        number of cluster.

        Parameters
        ----------
        features: np.array
            scaled and pca'ed feature-array (2D)

        Returns
        -------
        clusters: np.array
            in the order of self.inst_names, clusters of instances
        cluster_dict: Dict[int: int]
            maps clusters to indices of instances, i.e. for instances range(10):
            {0: [1,3,5], 1: [2,7,8], 2: [0,4,6,9]}
        """
        # get silhouette scores for k_means with 2 to 12 clusters
        scores = []
        for n_clusters in range(2, 12):
            km = KMeans(n_clusters=n_clusters)
            y_pred = km.fit_predict(features)
            score = silhouette_score(features, y_pred)
            scores.append(score)

        best_score = max(scores)
        best_run = scores.index(best_score)
        n_clusters = best_run + 2
        # cluster!
        km = KMeans(n_clusters=n_clusters)
        y_pred = km.fit_predict(features)

        self.logger.debug("%d Clusters: %s", n_clusters, str(y_pred))

        clusters = y_pred
        cluster_dict = {}
        for n in range(n_clusters):
            cluster_dict[n] = []
        for i, c in enumerate(y_pred):
            cluster_dict[c].append(self.inst_names[i])
        return y_pred, cluster_dict

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
        self.logger.debug("Calculating convex hulls for algorithm footprint.")
        if not insts: insts = self.inst_names
        hulls = []
        # For each cluster get convex hull
        for cluster in range(len(cluster_dict)):
            indexes_in_cluster = cluster_dict[cluster]
            instances = [inst for inst in self.inst_names
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

    def label_instances(self, conf:Configuration):
        """
        Returns dictionary with a label for each instance.

        Parameters
        ----------
            rh: RunHistory
                runhistory to take instances and costs from
            conf: Configuration
                configuration (=algorithm) for which to label the instances

        Returns
        -------
        labels: Dict[str:float]
            maps instance-names (strings) to label (floats)
        """
        cutoff = self.cutoff
        # Extract all instances
        insts = [i_s[0] for i_s in self.rh.get_runs_for_config(conf)]
        # Label all instances
        # For now: get costs and classify on those
        inst_cost = get_cost_dict_for_config(self.rh, conf)
        med = np.median(list(inst_cost.values()))
        result = {k:1 if v >= med else 0 for k,v in inst_cost.items()}
        self.logger.debug("Len(labels)=%d, len(insts)=%d",
                          len(result), len(self.inst_names))
        return result

        # Now for testing purposes, let's use a binary (solved=1,
        # unsolved=0)-labeling strategy
        inst_timeouts = get_timeout(self.rh, conf, self.cutoff)
        return inst_timeouts

    def plot_points(self, conf, out):
        """ Plot good versus bad for conf. Mainly for debugging labels!

        Parameters
        ----------
        conf: Configuration
            configuration for which to plot good vs bad
        out: str
            output path

        Returns
        -------
        outpath: str
            output path
        """
        labels = self.label_instances(conf)
        fig, ax = plt.subplots()

        good, bad = [], []
        for k, v in labels.items():
            point = self.features_2d[self.inst_names.index(k)]
            if v == 0:
                bad.append(point)
            else:
                good.append(point)
        good, bad = np.array(good), np.array(bad)
        ax.scatter(good[:, 0], good[:, 1], color="green")
        ax.scatter(bad[:, 0], bad[:, 1], color="red")
        ax.set_ylabel('Principal Component 1')
        ax.set_xlabel('Principal Component 2')
        fig.savefig(out)
        plt.close(fig)

        return out
