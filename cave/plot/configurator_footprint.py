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
import itertools
from collections import OrderedDict

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
import scipy as sp
import pandas as pd
import sklearn
from scipy.spatial.distance import hamming
from sklearn.manifold.mds import MDS
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from bokeh.plotting import figure, ColumnDataSource
from bokeh.embed import components
from bokeh.models import HoverTool, ColorBar, LinearColorMapper, BasicTicker
from bokeh.models.sources import CDSView
from bokeh.models.filters import GroupFilter

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

from cave.utils.convert_for_epm import convert_data_for_epm


class ConfiguratorFootprint(object):

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
            incumbents of best configurator run, last entry is final incumbent
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
        self.runs_per_rh = []  # number of configuration-evaluations per
                                  # rh (=configurator) in same order as
                                  # runhistories
        self.incs = incs
        self.max_plot = max_plot
        self.max_rhs_to_plot = 1  # Maximum number of runhistories 2 b plotted

        self.contour_step_size = contour_step_size
        self.relevant_rh = None
        self.output_dir = output_dir if output_dir else None

    def run(self):
        """
        Uses available Configurator-data to perform a MDS, estimate performance
        data and plot the configurator footprint.

        Returns
        -------
        html_code: str
            html-embedded plot-data
        """

        conf_matrix, conf_list, runs_per_conf = self.get_conf_matrix()
        self.logger.debug("Number of Configurations: %d" %
                         (conf_matrix.shape[0]))
        dists = self.get_distance(conf_matrix, self.scenario.cs)
        red_dists = self.get_mds(dists)

        contour_data = self.get_pred_surface(
                X_scaled=red_dists, conf_list=conf_list[:])

        inc_list = self.incs

        return self.plot(red_dists, conf_list, self.runs_per_rh,
                         inc_list=inc_list, contour_data=contour_data)

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
                if config in conf_list:
                    config_id, instance, seed = key
                    cost, time, status, additional_info = value
                    new_rh.add(config, cost, time, status, instance_id=instance,
                               seed=seed, additional_info=additional_info)
        self.relevant_rh = new_rh

        X, y, types = convert_data_for_epm(scenario=self.scenario,
                                           runhistory=new_rh)

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
        """
        Computes the distance between all pairs of configurations.

        Parameters
        ----------
        conf_matrx: np.array
            numpy array with cols as parameter values
        cs: ConfigurationSpace
            ConfigurationSpace to get conditionalities

        Returns
        -------
        dists: np.array
            np.array with distances between configurations i,j in dists[i,j] or dists[j,i]
        """
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
        """
        Get depth in configuration space of a given parameter name
        breadth search until reaching a leaf for the first time

        Parameters
        ----------
        cs: ConfigurationSpace
            ConfigurationSpace to get parents of a parameter
        param: str
            name of parameter to inspect
        """
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
        """
        Compute multi-dimensional scaling (using sklearn MDS) -- nonlinear scaling

        Parameters
        ----------
        dists: np.array
            full matrix of distances between all configurations

        Returns
        -------
        np.array
            scaled coordinates in 2-dim room
        """

        mds = MDS(
            n_components=2, dissimilarity="precomputed", random_state=12345)
        return mds.fit_transform(dists)

    def get_conf_matrix(self):
        """
        Iterates through runhistory to get a matrix of configurations (in
        vector representation), a list of configurations and the number of
        runs per configuration.

        Returns
        -------
        conf_matrix: np.array
            matrix of configurations in vector representation
        conf_list: list
            list of all Configuration objects that appeared in any runhistory
            the order of this list is used to determine all kinds of properties
            in the plotting
        runs_per_conf: np.array
            one-dim numpy array of runs per configuration
            FOR BEST RUNHISTORY ONLY
        """
        conf_matrix = []
        conf_list = []
        runs_runs_conf = []

        # Get all configurations. Index of c in conf_list serves as identifier
        for rh in self.runhistories:
            for c in rh.get_all_configs():
                if not c in conf_list:
                    conf_matrix.append(c.get_array())
                    conf_list.append(c)
        for inc in self.incs:
            if inc not in conf_list:
                conf_list.append(inc)

        # Get total runs per config per rh
        self.min_runs_per_conf = np.inf
        self.max_runs_per_conf = -np.inf
        for rh in self.runhistories:
            runs_per_conf = np.zeros(len(conf_list), dtype=int)
            for c in rh.get_all_configs():
                r_p_c = len(rh.get_runs_for_config(c))
                if r_p_c < self.min_runs_per_conf:
                    self.min_runs_per_conf = r_p_c
                elif r_p_c > self.max_runs_per_conf:
                    self.max_runs_per_conf = r_p_c
                runs_per_conf[conf_list.index(c)] = r_p_c
            self.runs_per_rh.append(np.array(runs_per_conf))

        self.logger.debug("Gathered %d configurations from %d runhistories." %
                          (len(conf_list), len(self.runs_per_rh)))

        return np.array(conf_matrix), conf_list, self.runs_per_rh[0]

    def _get_size(self, r_p_c):
        sizes = 5 + ((r_p_c - self.min_runs_per_conf) / (self.max_runs_per_conf - self.min_runs_per_conf)) * 20
        sizes *= np.array([0 if r == 0 else 1 for r in r_p_c])  # 0 size if 0 runs
        return sizes

    def _get_color(self, cds):
        """
        Parameters:
        -----------
        cds: ColumnDataSource
            data for bokeh plot

        Returns:
        --------
        colors: list
            list of color per config
        """
        colors = []
        for t in cds.data['type']:
            if t == "Default":
                colors.append('orange')
            elif "Incumbent" in  t:
                colors.append('red')
            else:
                colors.append('white')
        return colors

    def plot(self, X, conf_list: list, configurator_runs, runs_labels=None,
             inc_list: list=[], contour_data=None):
        """
        plots sampled configuration in 2d-space;
        saves results in self.output, if set

        Parameters
        ----------
        X: np.array
            np.array with 2-d coordinates for each configuration
        conf_list: list
            list of ALL configurations in the same order as X
        configurator_runs: list[np.array]
            list of configurator-runs to be analyzed, each as a np.array with
            the number of target-algorithm-runs per config.
            if 2 configurator-runs are analyzed with 3 configs evaluated in
            total, this might look like: [np.array([0,2,1]), np.array([2,7,1])]
        runs_labels: list[str]
            labels for the individual configurator-runs. if None, they are
            enumerated
        inc_list: list
            list of incumbents (Configuration)
        contour_data: list
            contour data (xx,yy,Z)

        Returns
        -------
        html_script: str
            HTML script representing the visualization
        """
        if not runs_labels:  # TODO We only plot first run atm anyway, this will be used later
            runs_labels = range(len(configurator_runs))

        hp_names = [k.name for k in  # Hyperparameter names
                    conf_list[0].configuration_space.get_hyperparameters()]

        # Create ColumnDataSource, x/y coordinates, config-params, sizes
        def escape_param_name(p):
            """Necessary because:
                1. parameters called 'size' or 'origin' might exist in cs
                2. '-' not allowed in bokeh's CDS"""
            return 'p_' + p.replace('-','_')

        source = ColumnDataSource(data=dict(x=X[:, 0], y=X[:, 1]))
        for k in hp_names:
            source.add([c[k] if c[k] else "None" for c in conf_list],
                       escape_param_name(k))
        # TODO differentiate between different configurator-runs below
        default = conf_list[0].configuration_space.get_default_configuration()
        conf_types = ["Default" if c == default else "Final Incumbent" if c == inc_list[-1]
                      else "Incumbent" if c in inc_list else "Candidate" for c in conf_list]
        # We group "Local Search" and "Random Search (sorted)" both into local
        origins = ["Unknown" if not c.origin else
                   "Acquisition Function" if c.origin.startswith("Local") or "sorted" in c.origin else
                   "Random" if c.origin.startswith("Random") else
                   "Unknown" for c in conf_list]
        source.add(conf_types, 'type')
        source.add(origins, 'origin')
        sizes = self._get_size(configurator_runs[0])
        sizes = [s * 3 if conf_types[idx] == "Default" else s for idx, s in enumerate(sizes)]
        source.add(sizes, 'size')
        source.add(self._get_color(source), 'color')
        source.add(configurator_runs[0], 'runs')

        # To enforce zorder, we categorize all entries according to their size
        # Since we plot all different zorder-levels sequentially, we use a
        # manually defined level of influence
        num_bins = 20  # How fine-grained the size-ordering should be
        min_size, max_size = min(source.data['size']), max(source.data['size'])
        step_size = (max_size - min_size) / num_bins
        zorder = [str(int((s - min_size) / step_size)) for s in source.data['size']]
        source.add(zorder, 'zorder')  # string, so we can apply group filter

        # Define what appears in tooltips
        # TODO add only important parameters (needs to change order of exec:
        #                                        pimp before conf-footprints)
        hover = HoverTool(tooltips=[('type', '@type'), ('origin', '@origin'), ('runs', '@runs')] +
                                   [(k, '@' + escape_param_name(k)) for k in hp_names])

        # bokeh-figure
        x_range = [min(X[:, 0]) - 1, max(X[:, 0]) + 1]
        y_range = [min(X[:, 1]) - 1, max(X[:, 1]) + 1]
        p = figure(plot_height=500, plot_width=600,
                   tools=[hover], x_range=x_range, y_range=y_range)

        # Plot contour
        if contour_data is not None:
            min_z = np.min(np.unique(contour_data[2]))
            max_z = np.max(np.unique(contour_data[2]))
            color_mapper = LinearColorMapper(palette="Viridis256",
                                             low=min_z, high=max_z)
            p.image(image=contour_data, x=x_range[0], y=y_range[0],
                    dw=x_range[1] - x_range[0], dh=y_range[1] - y_range[0],
                    color_mapper=color_mapper)
            color_bar = ColorBar(color_mapper=color_mapper,
                                 ticker=BasicTicker(desired_num_ticks=15),
                                 label_standoff=12,
                                 border_line_color=None, location=(0,0))
            color_bar.major_label_text_font_size = '12pt'
            p.add_layout(color_bar, 'right')

        # Scatter
        # TODO expand on multiple configurator runs(?)
        # Create views in order of plotting, so more interesting views are
        # plotted on top. Order of interest:
        # default > final-incumbent > incumbent > candidate
        #   local > random
        #     num_runs (ascending, more evaluated -> more interesting)

        def _get_marker(t, o):
            """ returns marker according to type t and origin o """
            if t == "Default":
                shape = 'triangle'
            elif t == 'Final Incumbent':
                shape = 'inverted_triangle'
            else:
                shape = 'square' if t == "Incumbent" else 'circle'
                shape += '_x' if o.startswith("Acquisition Function") else ''
            return shape

        views, markers = [], []
        for t in ['Candidate', 'Incumbent', 'Final Incumbent', 'Default']:
            for o in ['Unknown', 'Random', 'Acquisition Function']:
                for z in sorted(list(set(source.data['zorder'])),
                                key=lambda x: int(x)):
                    views.append(CDSView(source=source, filters=[
                            GroupFilter(column_name='type', group=t),
                            GroupFilter(column_name='origin', group=o),
                            GroupFilter(column_name='zorder', group=z)]))
                    markers.append(_get_marker(t, o))

        for view, marker in zip(views, markers):
            p.scatter(x='x', y='y',
                      source=source,
                      view=view,
                      color='color', line_color='black',
                      size='size',
                      marker=marker,
                      )

        p.xaxis.axis_label = "MDS-X"
        p.yaxis.axis_label = "MDS-Y"
        p.xaxis.axis_label_text_font_size = "15pt"
        p.yaxis.axis_label_text_font_size = "15pt"
        p.xaxis.major_label_text_font_size = "12pt"
        p.yaxis.major_label_text_font_size = "12pt"
        p.title.text_font_size = "15pt"
        p.legend.label_text_font_size = "15pt"

        script, div = components(p)

        return script, div
