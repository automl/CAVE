#!/bin/python

__author__ = "Marius Lindauer"
__copyright__ = "Copyright 2016, ML4AAD"
__license__ = "BSD"
__maintainer__ = "Marius Lindauer"
__email__ = "lindauer@cs.uni-freiburg.de"
__version__ = "0.0.1"

import os
import sys
import inspect
import logging
import json
import copy
import typing

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
import scipy as sp
import pandas as pd
import sklearn
from scipy.spatial.distance import hamming
from sklearn.manifold.mds import MDS
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

import mpld3

cmd_folder = os.path.realpath(
    os.path.abspath(os.path.split(inspect.getfile(inspect.currentframe()))[0]))
cmd_folder = os.path.realpath(os.path.join(cmd_folder, ".."))
if cmd_folder not in sys.path:
    sys.path.append(cmd_folder)

from smac.scenario.scenario import Scenario
from smac.runhistory.runhistory import RunHistory, DataOrigin
from smac.optimizer.objective import average_cost
from smac.epm.rf_with_instances import RandomForestWithInstances
from smac.configspace import ConfigurationSpace, Configuration
from smac.utils.util_funcs import get_types
from ConfigSpace.util import impute_inactive_values
from ConfigSpace.hyperparameters import FloatHyperparameter, IntegerHyperparameter
from ConfigSpace import CategoricalHyperparameter, UniformFloatHyperparameter, UniformIntegerHyperparameter

from cave.plot.confs_viz.utils.set_up import convert_data


class SampleViz(object):

    def __init__(self, scenario: Scenario,
                 runhistories: typing.List[RunHistory],
                 incs: list=None,
                 max_plot=None,
                 contour_step_size=0.2,
                 output_dir: str=None):
        '''
        Constructor

        Parameters
        ----------
        scenario: Scenario
            scenario
        runhistories: List[RunHistory]
            runhistories from configurator runs - first one assumed to be best
        incs: list
            incumbents (same length as runhistories!)
        max_plot: int
            maximum number of configs to plot
        contour_step_size: float
            step size of meshgrid to compute contour of fitness landscape
        output_dir: str
            output directory
        '''
        self.logger = logging.getLogger(
            self.__module__ + '.' + self.__class__.__name__)

        self.scenario = copy.deepcopy(scenario)  # pca changes feats
        self.runhistories = runhistories
        self.incs = incs
        self.max_plot = max_plot
        self.max_rhs_to_plot = 1  # Maximum number of runhistories 2 b plotted

        self.contour_step_size = contour_step_size
        if output_dir:
            self.output_dir = os.path.join(output_dir, 'conf_vizs.html')
        else:
            self.output_dir = None

    def run(self):
        '''
            main method
        '''

        conf_matrix, conf_list, runs_per_conf = self.get_conf_matrix()
        self.logger.info("Number of Configurations: %d" %
                         (conf_matrix.shape[0]))
        dists = self.get_distance(conf_matrix, self.scenario.cs)
        red_dists = self.get_mds(dists)

        contour_data = self.get_pred_surface(
                X_scaled=red_dists, conf_list=conf_list[:])

        inc_list = self.incs

        return self.plot(red_dists, conf_list, runs_per_conf,
                         inc_list, contour_data=contour_data)

    def get_pred_surface(self, X_scaled, conf_list: list):
        '''
            fit epm on the scaled input dimension and
            return data to plot a contour plot

            Parameters
            ----------
            X_scaled: np.array
                configurations in scaled 2dim
            conf_list: list
                list of Configuration objects

            Returns
            -------
            np.array, np.array, np.array
                x,y,Z for contour plots

        '''

        # use PCA to reduce features to also at most 2 dims
        n_feats = self.scenario.feature_array.shape[1]
        if n_feats > 2:
            self.logger.debug("Use PCA to reduce features to 2dim")
            insts = self.scenario.feature_dict.keys()
            feature_array = np.array([self.scenario.feature_dict[inst] for inst in insts])
            ss = StandardScaler()
            self.scenario.feature_array = ss.fit_transform(feature_array)
            pca = PCA(n_components=2)
            feature_array = pca.fit_transform(feature_array)
            n_feats = feature_array.shape[1]
            self.scenario.feature_array = feature_array
            self.scenario.feature_dict = dict([(inst, feature_array[idx,:]) for idx, inst in enumerate(insts)])
            self.scenario.n_features = 2

        # Create new rh with only wanted configs
        new_rh = RunHistory(average_cost)
        for rh in self.runhistories:
            for key, value in rh.data.items():
                config = rh.ids_config[key.config_id]
                if config in self.configs_to_plot:
                    config_id, instance, seed = key
                    cost, time, status, additional_info = value
                    new_rh.add(config, cost, time, status, instance_id=instance,
                               seed=seed, additional_info=additional_info)

        X, y, types = convert_data(scenario=self.scenario,
                                   runhistory=new_rh)
        self.logger.debug(X)

        types = np.array(np.zeros((2+n_feats)), dtype=np.uint)

        num_params = len(self.scenario.cs.get_hyperparameters())

        # impute missing values in configs
        conf_dict = {}
        for idx, c in enumerate(conf_list):
            conf_list[idx] = impute_inactive_values(c)
            conf_dict[str(conf_list[idx].get_array())] = X_scaled[idx, :]

        X_trans = []
        for x in X:
            x_scaled_conf = conf_dict[str(x[:num_params])]
            x_new = np.concatenate(
                        (x_scaled_conf, x[num_params:]), axis=0)
            X_trans.append(x_new)
        X_trans = np.array(X_trans)

        bounds = np.array([(0, np.nan), (0, np.nan)], dtype=object)
        model = RandomForestWithInstances(types=types, bounds=bounds,
                                          instance_features=np.array(self.scenario.feature_array),
                                          ratio_features=1.0)

        model.train(X_trans, y)

        self.logger.debug("RF fitted")

        plot_step = self.contour_step_size

        x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
        y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                             np.arange(y_min, y_max, plot_step))

        self.logger.debug("x_min: %f, x_max: %f, y_min: %f, y_max: %f" %(x_min, x_max, y_min, y_max))

        self.logger.debug("Predict on %d samples in grid to get surface" %(np.c_[xx.ravel(), yy.ravel()].shape[0]))
        Z, _ = model.predict_marginalized_over_instances(
            np.c_[xx.ravel(), yy.ravel()])

        Z = Z.reshape(xx.shape)

        return xx, yy, Z

    def get_distance(self, conf_matrix, cs: ConfigurationSpace):
        '''
            computes the distance between all pairs of configurations

            Parameters
            ----------
            conf_matrx: np.array
                numpy array with cols as parameter values
            cs: ConfigurationSpace
                ConfigurationSpace to get conditionalities

            Returns
            -------
            np.array with distances between configurations i,j in dists[i,j] or dists[j,i]
        '''
        n_confs = conf_matrix.shape[0]
        dists = np.zeros((n_confs, n_confs))

        is_cat = []
        depth = []
        for _, param in cs._hyperparameters.items():
            if type(param) == CategoricalHyperparameter:
                is_cat.append(True)
            else:
                is_cat.append(False)
            depth.append(self.get_depth(cs, param))
        is_cat = np.array(is_cat)
        depth = np.array(depth)

        for i in range(n_confs):
            for j in range(i + 1, n_confs):
                dist = np.abs(conf_matrix[i, :] - conf_matrix[j, :])
                dist[np.isnan(dist)] = 1
                dist[np.logical_and(is_cat, dist != 0)] = 1
                dist /= depth
                dists[i, j] = np.sum(dist)
                dists[j, i] = np.sum(dist)

        return dists

    def get_depth(self, cs: ConfigurationSpace, param: str):
        '''
            get depth in configuration space of a given parameter name
            breadth search until reaching a leaf for the first time

            Parameters
            ----------
            cs :ConfigurationSpace
                ConfigurationSpace to get parents of a parameter
            param: str
                name of parameter to inspect
        '''
        parents = cs.get_parents_of(param)
        if not parents:
            return 1
        new_parents = parents
        d = 1
        while new_parents:
            d += 1
            old_parents = new_parents
            new_parents = []
            for p in old_parents:
                pp = cs.get_parents_of(p)
                if pp:
                    new_parents.extend(pp)
                else:
                    return d

    def get_mds(self, dists):
        '''
            compute multi-dimensional scaling (using sklearn MDS) -- nonlinear scaling

            Parameters
            ----------
            dists: np.array
                full matrix of distances between all configurations

            Returns
            -------
            np.array
                scaled coordinates in 2-dim room
        '''

        mds = MDS(
            n_components=2, dissimilarity="precomputed", random_state=12345)
        return mds.fit_transform(dists)

    def get_conf_matrix(self):
        """Iterates through runhistory to get a matrix of configurations (in
        vector representation), a list of configurations and the number of
        runs per configuration.
        Does only consider configs in self.configs_to_plot.

        Parameters
        ----------
        rh: RunHistory
            runhistory of SMAC

        Returns
        -------
        np.array
            matrix of configurations in vector representation
        list
            list of Configuration objects
        np.array
            one-dim numpy array of runs per configuration
        """
        self.logger.debug("Gathering configurations to be plotted...")

        conf_matrix = []
        conf_list = []
        runs_runs_conf = []

        for rh in self.runhistories:
            for c in rh.get_all_configs():
                if not c in conf_list:
                    conf_matrix.append(c.get_array())
                    conf_list.append(c)

        # Get total runs per config
        for rh in self.runhistories:
            runs_per_conf = np.zeros(len(conf_list), dtype=int)
            for c in rh.get_all_configs():
                runs_per_conf[conf_list.index(c)] = len(rh.get_runs_for_config(c))
            runs_runs_conf.append(np.array(runs_per_conf))

        # Now decide what configurations to plot depending on max_plots and #runs
        ## Use #runs to determine the most "important" configs to plot
        #self.logger.info("Reducing number of configs (from %d) to be visualized"
        #                 ", plotting only the %d most often run configs.",
        #                 len(all_configs), len(configs_to_plot))
        runs_per_conf = np.zeros(len(conf_list), dtype=int)
        for r in runs_runs_conf:
            runs_per_conf += r
        assert(len(runs_per_conf) == len(conf_list))
        self.configs_to_plot = conf_list

        return np.array(conf_matrix), conf_list, runs_runs_conf

    def plot(self, X, conf_list: list, runs_runs_conf, inc_list: list,
             runs_labels=None, contour_data=None):
        '''
            plots sampled configuration in 2d-space;
            saves results in self.output, if set

            Parameters
            ----------
            X: np.array
                np.array with 2-d coordinates for each configuration
            conf_list: list
                list of configurations in the same order as X
            runs_runs_conf: list[np.array]
                list of runs to be analyzed, each a np.array with runs per
                config. if 2 runs are analyzed with 3 configs, this might look
                like: [np.array([0,2,1]), np.array([2,7,1])]
            inc_list: list
                list of incumbents (Configuration)
            contour_data: list
                contour data (xx,yy,Z)

            Returns
            -------
            html_script: str
                HTML script representing the visualization

        '''

        fig, ax = plt.subplots()

        if contour_data is not None:
            self.logger.debug("Plot Contour")
            min_z = np.min(np.unique(contour_data[2]))
            max_z = np.max(np.unique(contour_data[2]))
            v = np.linspace(min_z, max_z, 15, endpoint=True)
            contour = ax.contourf(contour_data[0], contour_data[1], contour_data[2],
                                  min(100, np.unique(contour_data[2]).shape[0]))
            plt.colorbar(contour, ticks=v[::-1])  #, pad=0.15)

        # Plot individual runs as scatter
        self.logger.debug("Plot Scatter")
        if not runs_labels:
            runs_labels = range(len(runs_runs_conf))
        for runs_per_conf, label in list(zip(runs_runs_conf,
                runs_labels))[:self.max_rhs_to_plot]:
            scatter = ax.scatter(
               X[:, 0], X[:, 1], sizes=np.log(runs_per_conf) + 10,
               color="white", edgecolors="black", label=label)

        ax.set_xlim(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5)
        ax.set_ylim(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5)

        inc_indx = []
        scatter_inc = None
        if inc_list:
            if isinstance(inc_list, list):
                inc_list = inc_list[:self.max_rhs_to_plot]
            else:
                inc_list = [inc_list]
            self.logger.debug("Plot Incumbents")
            for idx, conf in enumerate(conf_list):
                if conf in inc_list:
                    inc_indx.append(idx)
            self.logger.debug("Indexes of %d incumbent configurations: %s",
                              len(inc_list), str(inc_indx))
            scatter_inc = ax.scatter(X[inc_indx, 0],
                                     X[inc_indx, 1],
                                     color="black", edgecolors="white",
                                     sizes=np.log(runs_per_conf[inc_indx]) + 10)

        labels = []
        for idx, c in enumerate(conf_list):
            values = []
            names = []
            for p in c:
                if c[p]:
                    names.append(str(p))
                    values.append(c[p])

            label = pd.DataFrame(
                data=values, index=names, columns=["Conf %d" % (idx +
                    1)]).to_html()
            label = label.replace("dataframe", "config")
            labels.append(label)

        #self.logger.debug("Save test.png")
        #fig.savefig("test.png")

        # WORK IN PROGRESS
        # # Show only desired run
        # handles, labels = ax.get_legend_handles_labels() # return lines and labels
        # self.logger.debug("Handles: %s", handles)
        # self.logger.debug("Labels: %s", labels)
        # interactive_legend = mpld3.plugins.InteractiveLegendPlugin(zip(handles,
        #                                                          ax.collections),
        #                                                      labels,
        #                                                      alpha_unsel=0,
        #                                                      alpha_over=1,
        #                                                      start_visible=True)
        # mpld3.plugins.connect(fig, interactive_legend)

        tooltip = mpld3.plugins.PointHTMLTooltip(scatter, labels,
                                                 voffset=10, hoffset=10)#, css=self.css)

        mpld3.plugins.connect(fig, tooltip)

        if scatter_inc:
            tooltip = mpld3.plugins.PointHTMLTooltip(scatter_inc, np.array(labels)[inc_indx].tolist(),
                                                     voffset=10, hoffset=10)#, css=self.css)

        mpld3.plugins.connect(fig, tooltip)

        if self.output_dir:
            self.logger.debug("Save to %s", self.output_dir)
            with open(self.output_dir, "w") as fp:
                mpld3.save_html(fig, fp)

        html = mpld3.fig_to_html(fig)
        plt.close(fig)
        return html
