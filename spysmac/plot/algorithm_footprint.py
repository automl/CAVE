import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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
    def __init__(self, rh:RunHistory, inst_feat, cutoff, plotter=None):
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

        self.features = np.array([inst_feat[k] for k in self.inst_names])
        self.features_2d = self.reduce_dim(self.features)

        self.plotter = plotter
        self.cutoff = cutoff

    def reduce_dim(self, feature_array):
        """ Expects feature-array (not dict!) """
        # Perform PCA to reduce features to 2
        n_feats = feature_array.shape[1]
        if n_feats > 2:
            self.logger.debug("Use PCA to reduce features to two dimensions")
            ss = StandardScaler()
            feature_array = ss.fit_transform(feature_array)
            pca = PCA(n_components=2)
            feature_array = pca.fit_transform(feature_array)
        return feature_array

    def get_footprint(self, config):
        """ Calculate convex hull. """
        labels = label_instances(conf)
        label_list = [labels[i] for i in self.inst_names]


    def label_instances(self, conf:Configuration):
        """
        Returns dictionary with a label for each instance.

        Parameters
        ----------
            rh: RunHistory
                runhistory to take instances and costs from
            conf: Configuration
                configuration (=algorithm) for which to label the instances
        """
        cutoff = self.cutoff
        # Extract all instances
        insts = [i_s[0] for i_s in self.rh.get_runs_for_config(conf)]
        # Label all instances
        # For now: get costs and classify on those
        inst_cost = get_cost_dict_for_config(self.rh, conf)
        med = np.median(list(inst_cost.values()))
        return {k:1 if v >= med else 0 for k,v in inst_cost.items()}

        # Now for testing purposes, let's use a binary (solved=1,
        # unsolved=0)-labeling strategy
        inst_timeouts = get_timeout(self.rh, conf, self.cutoff)
        return inst_timeouts

    def plot_points(self, conf, out):
        """ Plot good versus bad for conf. Mainly for debug!"""
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
