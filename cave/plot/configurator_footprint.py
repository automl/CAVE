#!/bin/python

__author__ = "Marius Lindauer & Joshua Marben"
__copyright__ = "Copyright 2016, ML4AAD"
__license__ = "BSD"
__maintainer__ = "Joshua Marben"
__email__ = "marbenj@cs.uni-freiburg.de"
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

from bokeh.plotting import figure, ColumnDataSource, show
from bokeh.embed import components
from bokeh.models import HoverTool, ColorBar, LinearColorMapper, BasicTicker, CustomJS, Slider
from bokeh.models.sources import CDSView
from bokeh.models.filters import GroupFilter
from bokeh.layouts import row, column, widgetbox

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
from cave.utils.helpers import escape_parameter_name
from cave.utils.timing import timing
from cave.utils.io import export_bokeh


class ConfiguratorFootprint(object):

    def __init__(self, scenario: Scenario,
                 runhistory: RunHistory,
                 incs: list=None,
                 max_plot: int=5000,
                 contour_step_size=0.2,
                 output_dir: str=None,
                 time_slider: str='off',
                 num_quantiles: int=10,
                 ):
        '''
        Constructor

        Parameters
        ----------
        scenario: Scenario
            scenario
        runhistory: List[RunHistory]
            runhistory from configurator runs
        incs: list
            incumbents of best configurator run, last entry is final incumbent
        max_plot: int
            maximum number of configs to plot
        contour_step_size: float
            step size of meshgrid to compute contour of fitness landscape
        output_dir: str
            output directory
        time_slider: str
            one of ["off", "static", "prerender", "online"]
            prerender and online integrate a slider in the plot,
            static only renders a number of png's
            off only provides final interactive plot
        num_quantiles: int
            if time_slider is not off, defines the number of quantiles for the
            slider/ number of static pictures
        '''
        self.logger = logging.getLogger(
            self.__module__ + '.' + self.__class__.__name__)

        self.scenario = copy.deepcopy(scenario)  # pca changes feats
        self.rh = runhistory
        # runs_per_quantile holds a list for every quantile
        # of configuration-evaluations in the order of `conf-list`
        # so three configs and four quantiles thats:
        #   [[1, 2, 1], [3, 5, 2], [5, 6, 7], [9, 9, 8]]
        #   so to access the full # runs for the best configurator-run, just
        #   go for self.runs_per_quantile[-1]
        self.runs_per_quantile = []
        self.conf_list = []
        self.conf_matrix = []
        self.incs = incs
        self.max_plot = max_plot
        self.time_slider = time_slider

        self.num_quantiles = num_quantiles

        self.contour_step_size = contour_step_size
        self.output_dir = output_dir

    def run(self):
        """
        Uses available Configurator-data to perform a MDS, estimate performance
        data and plot the configurator footprint.
        """

        self.get_conf_matrix()
        self.logger.debug("Number of Configurations: %d", self.conf_matrix.shape[0])
        dists = self.get_distance(self.conf_matrix, self.scenario.cs)
        red_dists = self.get_mds(dists)

        contour_data = self.get_pred_surface(
                X_scaled=red_dists, conf_list=copy.deepcopy(self.conf_list))

        inc_list = self.incs

        return self.plot(red_dists, self.conf_list, self.runs_per_quantile,
                         inc_list=inc_list, contour_data=contour_data,
                         time_slider=self.time_slider)

    @timing
    def get_pred_surface(self, X_scaled, conf_list: list):
        """fit epm on the scaled input dimension and
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
        """

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

        self.logger.debug("Create new rh with relevant configs")
        new_rh = RunHistory(average_cost)
        for key, value in self.rh.data.items():
            config = self.rh.ids_config[key.config_id]
            if config in conf_list:
                config_id, instance, seed = key
                cost, time, status, additional_info = value
                new_rh.add(config, cost, time, status, instance_id=instance,
                           seed=seed, additional_info=additional_info)
        self.relevant_rh = new_rh

        self.logger.debug("Convert data for epm.")
        X, y, types = convert_data_for_epm(scenario=self.scenario,
                                           runhistory=new_rh,
                                           logger=self.logger)

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

        self.logger.debug("Train random forest for contour-plot.")
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

    @timing
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
        self.logger.debug("Calculate distance between configurations.")
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
            if i % (n_confs // 5) == 0:
                self.logger.debug("%.2f%% of all distances calculated...", 100 * i / n_confs)

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

    @timing
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
        # TODO n_jobs?
        mds = MDS(n_components=2, dissimilarity="precomputed", random_state=12345)
        return mds.fit_transform(dists)

    @timing
    def get_conf_matrix(self):
        """
        Iterates through runhistory to get a matrix of configurations (in
        vector representation), a list of configurations and the number of
        runs per configuration in a quantiled manner.

        Sideeffect creates as members
        conf_matrix: np.array
            matrix of configurations in vector representation
        conf_list: list
            list of all Configuration objects that appeared in runhistory
            the order of this list is used to determine all kinds of properties
            in the plotting (but is arbitrarily determined)
        runs_per_conf: np.array
            one-dim numpy array of runs per configuration
        """
        # Get all configurations. Index of c in conf_list serves as identifier
        for c in self.rh.get_all_configs():
            if not c in self.conf_list:
                self.conf_matrix.append(c.get_array())
                self.conf_list.append(c)
        for inc in self.incs:
            if inc not in self.conf_list:
                self.conf_matrix.append(inc.get_array())
                self.conf_list.append(inc)


        # We want to visualize the development over time, so we take
        # screenshots of the number of runs per config at different points
        # in (i.e. different quantiles of) the runhistory, LAST quantile
        # is full history!!
        runs_per_quantile = self._get_runs_per_config_quantiled(self.rh, quantiles=self.num_quantiles)

        # What configs to plot
        default = self.scenario.cs.get_default_configuration()
        self.logger.debug("Reducing number of configs from %d to %d, dropping from the fewest evaluations", len(self.conf_list), self.max_plot)
        keep_always = [self.conf_list.index(c) for c in self.incs + [default] if c in self.conf_list]  # Always plot default and incumbents
        keep_indices = sorted(range(len(runs_per_quantile[-1])), key=lambda x: runs_per_quantile[-1][x])[-self.max_plot:]
        keep_indices = list(set(keep_always + keep_indices))
        self.conf_list = np.array(self.conf_list)[keep_indices]
        self.conf_matrix = np.array(self.conf_matrix)[keep_indices]
        runs_per_quantile = [np.array(r_p_c)[keep_indices] for r_p_c in
                            runs_per_quantile]

        # Get minimum and maximum for sizes of dots
        self.min_runs_per_conf = min([i for i in runs_per_quantile[-1] if i > 0])
        self.max_runs_per_conf = max(runs_per_quantile[-1])

        self.logger.debug("Gathered %d configurations from 1 runhistories." % len(self.conf_list))
        self.conf_matrix = np.array(self.conf_matrix)

        self.runs_per_quantile = runs_per_quantile

    @timing
    def _get_runs_per_config_quantiled(self, rh, quantiles=10):
        """Creates a
        list of configurator-runs to be analyzed, each as a np.array with
        the number of target-algorithm-runs per config per quantile.
        two runhistories with three configs and four quantiles thats:
          [
           # runhistory 1
           [[1, 2, 1], [3, 5, 2], [5, 6, 7], [9, 9, 8]],
           # runhistory 2
           [[2, 5, 10], [4, 6, 13], [7, 8, 14], [9, 9, 14]],
          ]

        Parameters
        ----------
        rh: RunHistory
            rh to evaluate
        quantiles: int
            number of fractions to split rh into

        Returns:
        --------
        runs_per_config: Dict[Configuration : int]
            number of runs for config in rh up to given time
        """
        runs_total = len(rh.data)
        step = int(runs_total / quantiles)
        self.logger.debug("Creating %d quantiles with a step of %d and a total "
                          "runs of %d", quantiles, step, runs_total)
        r_p_q_p_c = []  # runs per quantile per config
        tmp_rh = RunHistory(average_cost)
        as_list = list(rh.data.items())
        ranges = [0] + list(range(step, runs_total-step, step)) + [runs_total]

        for i, j in zip(ranges[:-1], ranges[1:]):
            for idx in range(i, j):
                k, v = as_list[idx]
                tmp_rh.add(config=rh.ids_config[k.config_id],
                           cost=v.cost, time=v.time, status=v.status,
                           instance_id=k.instance_id, seed=k.seed)
            r_p_q_p_c.append([len(tmp_rh.get_runs_for_config(c)) for c in
                self.conf_list])
            #self.logger.debug("Using %d of %d runs", len(tmp_rh.data), len(rh.data))
        return r_p_q_p_c

    def _get_size(self, r_p_c):
        """
        Parameters
        ----------
        r_p_c: list[int]
            list with runs per config in order of self.conf_list

        Returns
        -------
        sizes: list[int]
            list with appropriate sizes for dots
        """
        self.logger.debug("Min runs per conf: %d, Max runs per conf: %d",
                          self.min_runs_per_conf, self.max_runs_per_conf)
        normalization_factor = self.max_runs_per_conf - self.min_runs_per_conf
        if normalization_factor == 0:  # All configurations same size
            normalization_factor = 1
        sizes = 5 + ((r_p_c - self.min_runs_per_conf) / normalization_factor) * 20
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

    def _plot_contour(self, p, contour_data, x_range, y_range):
        """Plot contour data.

        Parameters
        ----------
        p: bokeh.plotting.figure
            figure to be drawn upon
        contour_data: np.array
            array with contour data
        x_range: List[float, float]
            min and max of x-axis
        y_range: List[float, float]
            min and max of y-axis

        Returns
        -------
        p: bokeh.plotting.figure
            modified figure handle
        """
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
        return p

    def _plot_create_views(self, source):
        """Create views in order of plotting, so more interesting views are
        plotted on top. Order of interest:
        default > final-incumbent > incumbent > candidate
          local > random
            num_runs (ascending, more evaluated -> more interesting)
        Individual views are necessary, since bokeh can only plot one
        marker-type per 'scatter'-call

        Parameters
        ----------
        source: ColumnDataSource
            containing relevant information for plotting

        Returns
        -------
        views: List[CDSView]
            views in order of plotting
        markers: List[string]
            markers (to the view with the same index)
        """

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
        self.logger.debug("%d different glyph renderers, %d different zorder-values", len(views), len(set(source.data['zorder'])))
        return (views, markers)

    @timing
    def _plot_scatter(self, p, source, views, markers):
        """
        Parameters
        ----------
        p: bokeh.plotting.figure
            figure
        source: ColumnDataSource
            data container
        views: List[CDSView]
            list with views to be plotted (in order!)
        markers: List[str]
            corresponding markers to the views

        Returns
        -------
        scatter_handles: List[GlyphRenderer]
            glyph renderer per view
        """
        scatter_handles = []
        for view, marker in zip(views, markers):
            scatter_handles.append(p.scatter(x='x', y='y',
                                   source=source,
                                   view=view,
                                   color='color', line_color='black',
                                   size='size',
                                   marker=marker,
                                   ))

        p.xaxis.axis_label = "MDS-X"
        p.yaxis.axis_label = "MDS-Y"
        p.xaxis.axis_label_text_font_size = "15pt"
        p.yaxis.axis_label_text_font_size = "15pt"
        p.xaxis.major_label_text_font_size = "12pt"
        p.yaxis.major_label_text_font_size = "12pt"
        p.title.text_font_size = "15pt"
        p.legend.label_text_font_size = "15pt"
        self.logger.debug("Scatter-handles: %d", len(scatter_handles))
        return scatter_handles

    def _plot_get_source(self, conf_list, runs, X, inc_list, hp_names):
        """
        Create ColumnDataSource with all the necessary data
        Contains for each configuration evaluated on any run:

          - all parameters and values
          - origin (if conflicting, origin from best run counts)
          - type (default, incumbent or candidate)
          - # of runs
          - size
          - color

        Parameters
        ----------
        conf_list: list[Configuration]
            configurations
        runs: list[int]
            runs per configuration (same order as conf_list)
        X: np.array
            configuration-parameters as 2-dimensional array
        inc_list: list[Configuration]
            incumbents for this conf-run
        hp_names: list[str]
            names of hyperparameters

        Returns
        -------
        source: ColumnDataSource
            source with attributes as requested
        """
        source = ColumnDataSource(data=dict(x=X[:, 0], y=X[:, 1]))
        for k in hp_names:  # Add parameters for each config
            source.add([c[k] if c[k] else "None" for c in conf_list],
                       escape_parameter_name(k))
        default = conf_list[0].configuration_space.get_default_configuration()
        conf_types = ["Default" if c == default else "Final Incumbent" if c == inc_list[-1]
                      else "Incumbent" if c in inc_list else "Candidate" for c in conf_list]
        # We group "Local Search" and "Random Search (sorted)" both into local
        origins = [self._get_config_origin(c) for c in conf_list]
        source.add(conf_types, 'type')
        source.add(origins, 'origin')
        sizes = self._get_size(runs)
        sizes = [s * 3 if conf_types[idx] == "Default" else s for idx, s in enumerate(sizes)]
        source.add(sizes, 'size')
        source.add(self._get_color(source), 'color')
        source.add(runs, 'runs')
        # To enforce zorder, we categorize all entries according to their size
        # Since we plot all different zorder-levels sequentially, we use a
        # manually defined level of influence
        num_bins = 20  # How fine-grained the size-ordering should be
        min_size, max_size = min(source.data['size']), max(source.data['size'])
        step_size = (max_size - min_size) / num_bins
        zorder = [str(int((s - min_size) / step_size)) for s in source.data['size']]
        source.add(zorder, 'zorder')  # string, so we can apply group filter

        return source

    def _plot_get_callback_online(self, source):
        """Add an online timeslider. Difference to prerendered timeslider:
        information for all quantiles is already contained in source, callback
        updates the datasoure

        +: smaller files, that are quicker to load in browser
        -: updating data-source may take a long time

        Parameters
        ----------
        source: ColumnDataSource
            source contains runs1, runs2, runs3, ... and size1, size2, ... for
            the corresponding quantiles

        Returns
        -------
        time_slider: bokeh-Slider
            slider-widget, ready to register with plot
        """
        self.logger.debug("Create online time-slider!")
        code = """
var data = source.data;
var time = cb_obj.value;
data['runs'] = data['runs'+(time-1).toString()]
data['size'] = data['size'+(time-1).toString()]
source.change.emit();
"""
        # Create callback
        return CustomJS(args=dict(source=source), code=code)

    def _plot_get_callback_prerender(self, scatter_glyph_render_groups):
        """Add a prerendered timeslider. Difference to online timeslider:
        information for all quantiles is plotted in advance and only the
        relevant source is visible.

        +: faster interaction, no changes in data-source
        -: larger(!) files, that take longer to load into browser

        Parameters
        ----------
        scatter_glyph_render_groups: List[List[bokeh-glyphs]]
            list of lists, each sublist represents a quantile, a quantile is a
            list of all relevant glyphs in that quantile

        Returns
        -------
        time_slider: bokeh-Slider
            slider-widget, ready to register with plot
        """
        self.logger.debug("Create prerendered time-slider!")

        # Since runhistory doesn't contain a timestamp, but are ordered,
        # we use quantiles

        num_glyph_subgroups = sum([len(group) for group in scatter_glyph_render_groups])
        glyph_renderers_flattened = ['glyph_renderer' + str(i) for i in range(num_glyph_subgroups)]
        glyph_renderers = []
        start = 0
        for group in scatter_glyph_render_groups:
            glyph_renderers.append(glyph_renderers_flattened[start : start+len(group)])
            start += len(group)
        self.logger.debug("%d, %d, %d", len(scatter_glyph_render_groups),
                          len(glyph_renderers), num_glyph_subgroups)
        num_quantiles = len(scatter_glyph_render_groups)
        scatter_glyph_render_groups_flattened = [a for b in scatter_glyph_render_groups for a in b]
        args = {name : glyph for name, glyph in zip(glyph_renderers_flattened,
                                      scatter_glyph_render_groups_flattened)}
        code = "glyph_renderers = [" + ','.join(['[' + ','.join(group) + ']' for
                                    group in glyph_renderers]) + '];' + """
lab_len = cb_obj.end;
for (i = 0; i < lab_len; i++) {
    if (cb_obj.value == i + 1) {
        console.log('Setting to true: ' + i + '(' + glyph_renderers[i].length + ')')
        for (j = 0; j < glyph_renderers[i].length; j++) {
            glyph_renderers[i][j].visible = true;
            console.log('Setting to true: ' + i + ' : ' + j)
        }
    } else {
        console.log('Setting to false: ' + i + '(' + glyph_renderers[i].length + ')')
        for (j = 0; j < glyph_renderers[i].length; j++) {
            glyph_renderers[i][j].visible = false;
            console.log('Setting to false: ' + i + ' : ' + j)
        }
    }
}
"""
        callback = CustomJS(args=args, code=code)

        return callback

    def plot(self, X, conf_list: list, configurator_run,
             inc_list: list=None, contour_data=None, time_slider="off"):
        """
        plots sampled configuration in 2d-space;
        uses bokeh for interactive plot
        saves results in self.output, if set

        Parameters
        ----------
        X: np.array
            np.array with 2-d coordinates for each configuration
        conf_list: list
            list of ALL configurations in the same order as X
        configurator_run: list[np.array]
            configurator-runs to be analyzed, as a np.array with
            the number of target-algorithm-runs per config per quantile.
        inc_list: list
            list of incumbents (Configuration)
        contour_data: list
            contour data (xx,yy,Z)
        time_slider: str
            option to toggle how to implement time_slider.
            choose from: [off, prerender, online]
            where prerender creates a large file that might take some time to
            load and online creates a smaller file with long plot-updating times
            for large data

        Returns
        -------
        script: str
            script part of bokeh plot
        div: str
            div part of bokeh plot
        over_time_paths: List[str]
            list with paths to the different quantiled timesteps of the
            configurator run (for static evaluation)
        """
        if not time_slider in ['off', 'prerender', 'online', 'static']:
            raise ValueError("time_slider has to be one of ['off', 'prerender', 'online', 'static']")
        if not inc_list:
            inc_list = []
        over_time_paths = []  # developement of the search space over time

        best_run = configurator_run

        hp_names = [k.name for k in  # Hyperparameter names
                    conf_list[0].configuration_space.get_hyperparameters()]
        num_quantiles = len(best_run)

        # Get individual sources for quantiles of best run (first in list)
        sources = [self._plot_get_source(conf_list, quantiled_run, X, inc_list, hp_names)
                   for quantiled_run in best_run]

        # Define what appears in tooltips
        # TODO add only important parameters (needs to change order of exec pimp before conf-footprints)
        hover = HoverTool(tooltips=[('type', '@type'), ('origin', '@origin'), ('runs', '@runs')] +
                                   [(k, '@' + escape_parameter_name(k)) for k in hp_names])

        # bokeh-figure
        x_range = [min(X[:, 0]) - 1, max(X[:, 0]) + 1]
        y_range = [min(X[:, 1]) - 1, max(X[:, 1]) + 1]
        p = figure(plot_height=500, plot_width=600,
                   tools=[hover], x_range=x_range, y_range=y_range)

        # Plot contour
        if contour_data is not None:
           p = self._plot_contour(p, contour_data, x_range, y_range)
        scatter_glyph_render_groups = []

        # If timeslider should update the data online, we need to add
        # information about each quantile to the "base"-source
        if time_slider == 'online':
            for idx, q in enumerate(best_run):
                sources[-1].add(q, 'runs' + str(idx))
                sizes = [s * 3 if sources[-1].data['type'][idx] == "Default" else s for idx, s
                         in enumerate(self._get_size(q))]
                sources[-1].add(sizes, 'size' + str(idx))

        # If time_slider is not prerender, we don't want all the sources in
        # the plot -> in that case we create a new plot for the individual
        # quantile-png-exports.
        if not (time_slider == 'prerender'):
            p_quantiles = figure(plot_height=500, plot_width=600,
                                 tools=[hover], x_range=x_range, y_range=y_range)
            if contour_data is not None:
               p_quantiles = self._plot_contour(p_quantiles, contour_data, x_range, y_range)
        else:
            p_quantiles = p  # if prerender, save all sources in original plot

        for idx, source in enumerate(sources):
            if idx == len(sources) - 1:  # final view on original plot!!
                p_quantiles = p
            elif time_slider == 'off':  # skip all others if slider is off
                continue
            views, markers = self._plot_create_views(source)
            self.logger.debug("Plotting quantile %d!", idx)
            scatter_glyph_render_groups.append(self._plot_scatter(p_quantiles, source, views, markers))
            if self.output_dir:
                file_path = "content/images/cfp_over_time/configurator_footprint" + str(idx) + ".png"
                over_time_paths.append(os.path.join(self.output_dir, file_path))
                self.logger.debug("Saving plot to %s", over_time_paths[-1])
                export_bokeh(p_quantiles, over_time_paths[-1], self.logger)

        if time_slider in ['off', 'static']:
            layout = column(p)
        else:
            # Slider below plot
            if time_slider == 'prerender':
                callback = self._plot_get_callback_prerender(scatter_glyph_render_groups)
            elif time_slider == 'online':
                callback = self._plot_get_callback_online(sources[-1])
            slider = Slider(start=1, end=num_quantiles,
                            value=num_quantiles, step=1,
                            callback=callback, title='Time')
            layout = column(p, widgetbox(slider))

        script, div = components(layout)

        if self.output_dir:
            path = os.path.join(self.output_dir, "content/images/configurator_footprint.png")
            export_bokeh(p, path, self.logger)

        return script, div, over_time_paths

    def _get_config_origin(self, c):
        """Return appropriate configuration origin

        Parameters
        ----------
        c: Configuration
            configuration to be examined

        Returns
        -------
        origin: str
            origin of configuration (e.g. "Local", "Random", etc.)
        """
        if not c.origin:
            origin = "Unknown"
        elif c.origin.startswith("Local") or "sorted" in c.origin:
            origin = "Acquisition Function"
        elif c.origin.startswith("Random"):
            origin = "Random"
        else:
            origin = "Unknown"
        return origin
