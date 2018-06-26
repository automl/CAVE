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
import copy
import time

import numpy as np
from sklearn.manifold.mds import MDS
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from bokeh.plotting import figure, ColumnDataSource
from bokeh.embed import components
from bokeh.models import HoverTool, ColorBar, LinearColorMapper, BasicTicker, CustomJS, Slider
from bokeh.models.sources import CDSView
from bokeh.models.filters import GroupFilter
from bokeh.layouts import column, widgetbox

cmd_folder = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile(inspect.currentframe()))[0]))  # noqa
cmd_folder = os.path.realpath(os.path.join(cmd_folder, ".."))  # noqa
if cmd_folder not in sys.path:  # noqa
    sys.path.append(cmd_folder)  # noqa

from smac.scenario.scenario import Scenario
from smac.runhistory.runhistory import RunHistory
from smac.optimizer.objective import average_cost
from smac.epm.rf_with_instances import RandomForestWithInstances
from smac.configspace import ConfigurationSpace
from ConfigSpace.util import impute_inactive_values
from ConfigSpace import CategoricalHyperparameter

from cave.utils.convert_for_epm import convert_data_for_epm
from cave.utils.helpers import escape_parameter_name
from cave.utils.timing import timing
from cave.utils.io import export_bokeh


class ConfiguratorFootprint(object):

    def __init__(self,
                 scenario: Scenario,
                 rh: RunHistory,
                 incs: list=None,
                 max_plot: int=-1,
                 contour_step_size=0.2,
                 output_dir: str=None,
                 time_slider: bool=False,
                 num_quantiles: int=10,
                 ):
        '''
        Constructor

        Parameters
        ----------
        scenario: Scenario
            scenario
        rh: RunHistory
            runhistory from configurator run, only runs during optimization
        incs: list
            incumbents of best configurator run, last entry is final incumbent
        max_plot: int
            maximum number of configs to plot, if -1 plot all
        contour_step_size: float
            step size of meshgrid to compute contour of fitness landscape
        output_dir: str
            output directory
        time_slider: bool
            whether or not to have a time_slider-widget on cfp-plot
            INCREASES FILE-SIZE DRAMATICALLY
        num_quantiles: int
            number of quantiles for the slider/ number of static pictures
        '''
        self.logger = logging.getLogger(self.__module__ + '.' + self.__class__.__name__)

        self.scenario = scenario
        self.orig_rh = rh
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
        default = self.scenario.cs.get_default_configuration()
        self.orig_rh = self.reduce_runhistory(self.orig_rh, self.max_plot, keep=self.incs+[default])
        conf_matrix, conf_list, runs_per_quantile = self.get_conf_matrix(self.orig_rh, self.incs)
        self.logger.debug("Number of Configurations: %d", conf_matrix.shape[0])
        dists = self.get_distance(conf_matrix, self.scenario.cs)
        red_dists = self.get_mds(dists)

        contour_data = self.get_pred_surface(self.orig_rh, X_scaled=red_dists,
                                             conf_list=copy.deepcopy(conf_list),
                                             contour_step_size=self.contour_step_size)

        return self.plot(red_dists,
                         conf_list,
                         runs_per_quantile,
                         inc_list=self.incs,
                         contour_data=contour_data,
                         time_slider=self.time_slider)

    @timing
    def get_pred_surface(self, rh, X_scaled, conf_list: list, contour_step_size):
        """fit epm on the scaled input dimension and
        return data to plot a contour plot of the empirical performance

        Parameters
        ----------
        rh: RunHistory
            runhistory
        X_scaled: np.array
            configurations in scaled 2dim
        conf_list: list
            list of Configuration objects

        Returns
        -------
        contour_data: (np.array, np.array, np.array)
            x, y, Z for contour plots
        """
        # use PCA to reduce features to also at most 2 dims
        scen = copy.deepcopy(self.scenario)  # pca changes feats
        if scen.feature_array.shape[1] > 2:
            self.logger.debug("Use PCA to reduce features to from %d dim to 2 dim", scen.feature_array.shape[1])
            # perform PCA
            insts = scen.feature_dict.keys()
            feature_array = np.array([scen.feature_dict[i] for i in insts])
            feature_array = StandardScaler().fit_transform(feature_array)
            feature_array = PCA(n_components=2).fit_transform(feature_array)
            # inject in scenario-object
            scen.feature_array = feature_array
            scen.feature_dict = dict([(inst, feature_array[idx, :]) for idx, inst in enumerate(insts)])
            scen.n_features = 2

        # convert the data to train EPM on 2-dim featurespace (for contour-data)
        self.logger.debug("Convert data for epm.")
        X, y, types = convert_data_for_epm(scenario=scen, runhistory=rh, logger=self.logger)
        types = np.array(np.zeros((2 + scen.feature_array.shape[1])), dtype=np.uint)
        num_params = len(scen.cs.get_hyperparameters())

        # impute missing values in configs and insert MDS'ed (2dim) configs to the right positions
        conf_dict = {}
        for idx, c in enumerate(conf_list):
            conf_list[idx] = impute_inactive_values(c)
            conf_dict[str(conf_list[idx].get_array())] = X_scaled[idx, :]

        X_trans = []
        for x in X:
            x_scaled_conf = conf_dict[str(x[:num_params])]
            # append scaled config + pca'ed features (total of 4 values) per config/feature-sample
            X_trans.append(np.concatenate((x_scaled_conf, x[num_params:]), axis=0))
        X_trans = np.array(X_trans)

        self.logger.debug("Train random forest for contour-plot.")
        bounds = np.array([(0, np.nan), (0, np.nan)], dtype=object)
        model = RandomForestWithInstances(types=types, bounds=bounds,
                                          instance_features=np.array(scen.feature_array),
                                          ratio_features=1.0)

        start = time.time()
        model.train(X_trans, y)
        self.logger.debug("Fitting random forest took %f time", time.time() - start)

        x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
        y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, contour_step_size),
                             np.arange(y_min, y_max, contour_step_size))

        self.logger.debug("x_min: %f, x_max: %f, y_min: %f, y_max: %f", x_min, x_max, y_min, y_max)
        self.logger.debug("Predict on %d samples in grid to get surface (step-size: %f)",
                          np.c_[xx.ravel(), yy.ravel()].shape[0], contour_step_size)

        start = time.time()
        Z, _ = model.predict_marginalized_over_instances(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        self.logger.debug("Predicting random forest took %f time", time.time() - start)

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
        # TODO there are ways to extend MDS to provide a transform-method. if
        #   available, train on randomly sampled configs and plot all
        # TODO MDS provides 'n_jobs'-argument for parallel computing...
        mds = MDS(n_components=2, dissimilarity="precomputed", random_state=12345)
        dists = mds.fit_transform(dists)
        self.logger.debug("MDS-stress: %f", mds.stress_)
        return dists

    def reduce_runhistory(self,
                          rh: RunHistory,
                          max_configs: int,
                          keep=None):
        """
        Reduce configs to desired number, by default just drop the configs with the
        fewest runs.

        Parameters
        ----------
        rh: RunHistory
            runhistory that is to be reduced
        max_configs: int
            if > -1 reduce runhistory to at most max_configs
        keep: List[Configuration]
            list of configs that should be kept for sure (e.g. default,
            incumbents)

        Returns
        -------
        rh: RunHistory
            reduced runhistory
        """
        configs = rh.get_all_configs()
        if max_configs <= 0 or max_configs > len(configs):  # keep all
            return rh

        runs = [(c, len(rh.get_runs_for_config(c))) for c in configs]
        if not keep:
            keep = []
        runs = sorted(runs, key=lambda x: x[1])[-self.max_plot:]
        keep = [r[0] for r in runs] + keep
        self.logger.info("Reducing number of configs from %d to %d, dropping from the fewest evaluations",
                         len(configs), len(keep))

        new_rh = RunHistory(average_cost)
        for k, v in list(rh.data.items()):
            c = rh.ids_config[k.config_id]
            if c in keep:
                new_rh.add(config=rh.ids_config[k.config_id],
                           cost=v.cost, time=v.time, status=v.status,
                           instance_id=k.instance_id, seed=k.seed)
        return new_rh

    @timing
    def get_conf_matrix(self, rh, incs):
        """
        Iterates through runhistory to get a matrix of configurations (in
        vector representation), a list of configurations and the number of
        runs per configuration in a quantiled manner.

        Parameters
        ----------
        rh: RunHistory
            smac.runhistory
        incs: List[Configuration]
            incumbents of this configurator run, last entry is final incumbent

        Returns
        -------
        conf_matrix: np.array
            matrix of configurations in vector representation
        conf_list: np.array
            list of all Configuration objects that appeared in runhistory
            the order of this list is used to determine all kinds of properties
            in the plotting (but is arbitrarily determined)
        runs_per_quantile: np.array
            numpy array of runs per configuration per quantile
        """
        # Get all configurations. Index of c in conf_list serves as identifier
        conf_list = []
        conf_matrix = []
        for c in rh.get_all_configs():
            if c not in conf_list:
                conf_matrix.append(c.get_array())
                conf_list.append(c)
        for inc in incs:
            if inc not in conf_list:
                conf_matrix.append(inc.get_array())
                conf_list.append(inc)

        # We want to visualize the development over time, so we take
        # screenshots of the number of runs per config at different points
        # in (i.e. different quantiles of) the runhistory, LAST quantile
        # is full history!!
        runs_per_quantile = self._get_runs_per_config_quantiled(rh, conf_list, quantiles=self.num_quantiles)
        assert(len(runs_per_quantile) == self.num_quantiles)

        # Get minimum and maximum for sizes of dots
        self.min_runs_per_conf = min([i for i in runs_per_quantile[-1] if i > 0])
        self.max_runs_per_conf = max(runs_per_quantile[-1])
        self.logger.debug("Min runs per conf: %d, Max runs per conf: %d",
                          self.min_runs_per_conf, self.max_runs_per_conf)

        self.logger.debug("Gathered %d configurations from 1 runhistories." % len(conf_list))

        runs_per_quantile = np.array([np.array(run) for run in runs_per_quantile])
        return np.array(conf_matrix), np.array(conf_list), runs_per_quantile

    @timing
    def _get_runs_per_config_quantiled(self, rh, conf_list, quantiles):
        """Returns a list of lists, each sublist representing the current state
        at that timestep (quantile). The current state means a list of times
        each config was evaluated at that timestep.

        Parameters
        ----------
        rh: RunHistory
            rh to evaluate
        conf_list: list
            list of all Configuration objects that appeared in runhistory
        quantiles: int
            number of fractions to split rh into

        Returns:
        --------
        runs_per_quantile: np.array
            numpy array of runs per configuration per quantile
        """
        runs_total = len(rh.data)
        # Create LINEAR ranges. TODO do we want log? -> this line
        ranges = [int(r) for r in np.linspace(0, runs_total, quantiles + 1)]
        self.logger.debug("Creating %d quantiles with a step of %.2f and a total "
                          "runs of %d", quantiles, runs_total/quantiles, runs_total)
        self.logger.debug("Ranges: %s", str(ranges))

        # Iterate over the runhistory's entries in ranges and creating each
        # sublist from a "snapshot"-runhistory
        r_p_q_p_c = []  # runs per quantile per config
        as_list = list(rh.data.items())
        tmp_rh = RunHistory(average_cost)
        for i, j in zip(ranges[:-1], ranges[1:]):
            for idx in range(i, j):
                k, v = as_list[idx]
                tmp_rh.add(config=rh.ids_config[k.config_id],
                           cost=v.cost, time=v.time, status=v.status,
                           instance_id=k.instance_id, seed=k.seed)
            r_p_q_p_c.append([len(tmp_rh.get_runs_for_config(c)) for c in conf_list])
        return r_p_q_p_c

    def _get_size(self, r_p_c):
        """Returns size of scattered points in dependency of runs per config

        Parameters
        ----------
        r_p_c: list[int]
            list with runs per config in order of self.conf_list

        Returns
        -------
        sizes: list[int]
            list with appropriate sizes for dots
        """
        normalization_factor = self.max_runs_per_conf - self.min_runs_per_conf
        min_size, enlargement_factor = 5, 20
        if normalization_factor == 0:  # All configurations same size
            normalization_factor = 1
        sizes = min_size + ((r_p_c - self.min_runs_per_conf) / normalization_factor) * enlargement_factor
        sizes *= np.array([0 if r == 0 else 1 for r in r_p_c])  # 0 size if 0 runs
        return sizes

    def _get_color(self, types):
        """Determine appropriate color for all configurations

        Parameters:
        -----------
        types: List[str]
            type of configuration

        Returns:
        --------
        colors: list
            list of color per config
        """
        colors = []
        for t in types:
            if t == "Default":
                colors.append('orange')
            elif "Incumbent" in t:
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
                             border_line_color=None, location=(0, 0))
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
        marker-typei (circle, triangle, ...) per 'scatter'-call

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
        self.logger.debug("%d different glyph renderers, %d different zorder-values",
                          len(views), len(set(source.data['zorder'])))
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
        # Remove all configurations without any runs
        keep = [i for i in range(len(runs)) if runs[i] > 0]
        runs = np.array(runs)[keep]
        conf_list = np.array(conf_list)[keep]
        X = X[keep]

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
        source.add(self._get_color(source.data['type']), 'color')
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

    def _plot_get_timeslider(self, scatter_glyph_render_groups):
        """Add a prerendered timeslider.
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

        # Since runhistory doesn't contain a timestamp, but are ordered, we use quantiles
        num_glyph_subgroups = sum([len(group) for group in scatter_glyph_render_groups])
        glyph_renderers_flattened = ['glyph_renderer' + str(i) for i in range(num_glyph_subgroups)]
        glyph_renderers = []
        start = 0
        for group in scatter_glyph_render_groups:
            glyph_renderers.append(glyph_renderers_flattened[start: start+len(group)])
            start += len(group)
        self.logger.debug("%d, %d, %d", len(scatter_glyph_render_groups),
                          len(glyph_renderers), num_glyph_subgroups)
        scatter_glyph_render_groups_flattened = [a for b in scatter_glyph_render_groups for a in b]
        args = {name: glyph for name, glyph in zip(glyph_renderers_flattened,
                                                   scatter_glyph_render_groups_flattened)}
        code = "glyph_renderers = [" + ','.join(['[' + ','.join(group) + ']' for
                                                 group in glyph_renderers]) + '];' + """
lab_len = cb_obj.end;
for (i = 0; i < lab_len; i++) {
    if (cb_obj.value == i + 1) {
        // console.log('Setting to true: ' + i + '(' + glyph_renderers[i].length + ')')
        for (j = 0; j < glyph_renderers[i].length; j++) {
            glyph_renderers[i][j].visible = true;
            // console.log('Setting to true: ' + i + ' : ' + j)
        }
    } else {
        // console.log('Setting to false: ' + i + '(' + glyph_renderers[i].length + ')')
        for (j = 0; j < glyph_renderers[i].length; j++) {
            glyph_renderers[i][j].visible = false;
            // console.log('Setting to false: ' + i + ' : ' + j)
        }
    }
}
"""
        callback = CustomJS(args=args, code=code)
        num_quantiles = len(scatter_glyph_render_groups)
        slider = Slider(start=1, end=num_quantiles,
                        value=num_quantiles, step=1,
                        callback=callback, title='Time')
        return slider

    def plot(self,
             X,
             conf_list: list,
             runs_per_quantile,
             inc_list: list=None,
             contour_data=None,
             time_slider=False):
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
        runs_per_quantile: list[np.array]
            configurator-run to be analyzed, as a np.array with
            the number of target-algorithm-runs per config per quantile.
        inc_list: list
            list of incumbents (Configuration)
        contour_data: list
            contour data (xx,yy,Z)
        time_slider: bool
            whether or not to have a time_slider-widget on cfp-plot
            INCREASES FILE-SIZE DRAMATICALLY

        Returns
        -------
        (script, div): str
            script and div of the bokeh-figure
        over_time_paths: List[str]
            list with paths to the different quantiled timesteps of the
            configurator run (for static evaluation)
        """
        if not inc_list:
            inc_list = []
        over_time_paths = []  # development of the search space over time

        hp_names = [k.name for k in  # Hyperparameter names
                    conf_list[0].configuration_space.get_hyperparameters()]

        # Get individual sources for quantiles
        sources = [self._plot_get_source(conf_list, quantiled_run, X, inc_list, hp_names)
                   for quantiled_run in runs_per_quantile]

        # Define what appears in tooltips
        # TODO add only important parameters (needs to change order of exec pimp before conf-footprints)
        hover = HoverTool(tooltips=[('type', '@type'), ('origin', '@origin'), ('runs', '@runs')] +
                                   [(k, '@' + escape_parameter_name(k)) for k in hp_names])

        # bokeh-figure
        x_range = [min(X[:, 0]) - 1, max(X[:, 0]) + 1]
        y_range = [min(X[:, 1]) - 1, max(X[:, 1]) + 1]

        scatter_glyph_render_groups = []
        for idx, source in enumerate(sources):
            if not time_slider or idx == 0:
                # Only plot all quantiles in one plot if timeslider is on
                p = figure(plot_height=500, plot_width=600,
                           tools=[hover, 'save'], x_range=x_range, y_range=y_range)
                if contour_data is not None:
                    p = self._plot_contour(p, contour_data, x_range, y_range)
            views, markers = self._plot_create_views(source)
            self.logger.debug("Plotting quantile %d!", idx)
            scatter_glyph_render_groups.append(self._plot_scatter(p, source, views, markers))
            if self.output_dir:
                file_path = "content/images/cfp_over_time/configurator_footprint" + str(idx) + ".png"
                over_time_paths.append(os.path.join(self.output_dir, file_path))
                self.logger.debug("Saving plot to %s", over_time_paths[-1])
                export_bokeh(p, over_time_paths[-1], self.logger)

        if time_slider:
            self.logger.debug("Adding timeslider")
            slider = self._plot_get_timeslider(scatter_glyph_render_groups)
            layout = column(p, widgetbox(slider))
        else:
            self.logger.debug("Not adding timeslider")
            layout = column(p)

        script, div = components(layout)

        if self.output_dir:
            path = os.path.join(self.output_dir, "content/images/configurator_footprint.png")
            export_bokeh(p, path, self.logger)

        return (script, div), over_time_paths

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
