import sys
import os
import logging

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from typing import Union, List, Dict
plt.style.use(os.path.join(os.path.dirname(__file__), 'mpl_style'))
from matplotlib import ticker

from ConfigSpace.util import impute_inactive_values

from smac.utils.validate import Validator
from smac.configspace import Configuration, convert_configurations_to_array
from smac.epm.rf_with_instances import RandomForestWithInstances
from smac.optimizer.objective import average_cost, _cost
from smac.runhistory.runhistory2epm import RunHistory2EPM4Cost
from smac.utils.util_funcs import get_types
from smac.runhistory.runhistory import RunHistory
from smac.utils.validate import Validator

from cave.reader.configurator_run import ConfiguratorRun
from cave.plot.scatter import plot_scatter_plot
from cave.plot.configurator_footprint import ConfiguratorFootprint
from cave.plot.parallel_coordinates import ParallelCoordinatesPlotter
from cave.utils.helpers import get_cost_dict_for_config, get_timeout
from cave.utils.io import export_bokeh

from bokeh.plotting import figure, ColumnDataSource
from bokeh.embed import components
from bokeh.models import HoverTool, Range1d, FuncTickFormatter
from bokeh.models.sources import CDSView
from bokeh.models.filters import GroupFilter
from bokeh.models.tickers import AdaptiveTicker

__author__ = "Joshua Marben"
__copyright__ = "Copyright 2017, ML4AAD"
__license__ = "3-clause BSD"
__maintainer__ = "Joshua Marben"
__email__ = "joshua.marben@neptun.uni-freiburg.de"

class Plotter(object):
    """
    This class is used to outsource some plotting routines and some of the parts
    of analysis that require lots of plot-related code (such as conf_viz or
    parallel_coordinates). It should be invoked via the Analyzer-class. More
    complicated or generalized plotting routines are outsourced and imported, so
    they can be easily adapted into other projects.
    """

    def __init__(self, scenario, output_dir):
        """
        Parameters
        ----------
        scenario: Scenario
            scenario to take cutoff, train- test, etc from
        train_test: bool
            whether to make a distinction between train and test
        conf1_runs, conf2_runs: list(RunValue)
            lists with smac.runhistory.runhistory.RunValue, from which to read
            cost or time
        output_dir: str
            output-directory
        """
        self.logger = logging.getLogger("cave.plotter")
        self.scenario = scenario
        self.train_test = len(self.scenario.train_insts) > 1 and len(self.scenario.test_insts) > 1
        self.output_dir = output_dir
        self.configurator_footprint_rh = None  # Can be initialized from extern

    def plot_scatter(self, default, incumbent, runhistory):
        """
        Creates a scatterplot of the two configurations on the given set of
        instances.
        Saves plot to file.

        Parameters:
        -----------
        output_fn_base: string
            base-path to save plot to
        """
        #TODO docstring
        self.logger.debug("Plot scatter to %s[train|test].png",
                os.path.join(self.output_dir, 'scatter'))

        metric = self.scenario.run_obj
        timeout = self.scenario.cutoff
        labels = ["default {}".format(self.scenario.run_obj), "incumbent {}".format(self.scenario.run_obj)]

        default_train = np.array([v for k, v in
                                  get_cost_dict_for_config(runhistory, default).items()
                                  if k in self.scenario.train_insts])
        incumbent_train = np.array([v for k, v in
                                   get_cost_dict_for_config(runhistory, incumbent).items()
                                   if k in self.scenario.train_insts])
        min_val = min(min(default_train),
                      min(incumbent_train))
        if self.train_test:
            default_test = np.array([v for k, v in
                                    get_cost_dict_for_config(runhistory, default).items()
                                    if k in self.scenario.test_insts])
            incumbent_test = np.array([v for k, v in
                                      get_cost_dict_for_config(runhistory, incumbent).items()
                                      if k in self.scenario.test_insts])
            min_val = min(min_val, min(default_test),
                                   min(incumbent_test))

        paths = os.path.join(self.output_dir, 'scatter_train.png')

        fig = plot_scatter_plot((default_train,), (incumbent_train,),
                                labels, metric=metric,
                                user_fontsize=22,
                                min_val=min_val,
                                max_val=timeout,
                                jitter_timeout=True)
        fig.savefig(paths)
        plt.close(fig)
        if self.train_test:
            paths = [paths, os.path.join(self.output_dir, 'scatter_test.png')]
            fig = plot_scatter_plot((default_test,), (incumbent_test,),
                                    labels, metric=metric,
                                    user_fontsize=22,
                                    min_val=min_val,
                                    max_val=timeout,
                                    jitter_timeout=True)
            fig.savefig(paths[-1])
            plt.close(fig)
        return paths

    def plot_cdf_compare(self, default, incumbent, runhistory):
        """
        Plot the cumulated distribution functions for given configurations,
        plots will share y-axis and if desired x-axis.
        Saves plot to file.

        Parameters
        ----------
        output: List[str]
            filename, default: CDF_compare.png
        """
        self.logger.debug("Plot CDF to %s_[train|test].png",
                          os.path.join(self.output_dir, 'cdf'))

        timeout = self.scenario.cutoff

        def prepare_data(x_data):
            """ Helper function to keep things easy, generates y_data and
            manages x_data-timeouts """
            x_data = sorted(x_data)
            y_data = np.array(range(len(x_data)))/(len(x_data)-1)
            for idx in range(len(x_data)):
                if (timeout != None) and (x_data[idx] >= timeout):
                    x_data[idx] = timeout
                    y_data[idx] = y_data[idx-1]
            return (x_data, y_data)

        # Generate y_data
        data = {'default' :
                 {'train' : prepare_data(
                           np.array([v for k, v in
                                    get_cost_dict_for_config(runhistory, default).items()
                                    if k in self.scenario.train_insts])),
                  'test' : prepare_data(
                           np.array([v for k, v in
                                    get_cost_dict_for_config(runhistory, default).items()
                                    if k in self.scenario.test_insts]))},
                'incumbent' :
                 {'train' : prepare_data(
                           np.array([v for k, v in
                                    get_cost_dict_for_config(runhistory, incumbent).items()
                                    if k in self.scenario.train_insts])),
                  'test' : prepare_data(
                           np.array([v for k, v in
                                    get_cost_dict_for_config(runhistory, incumbent).items()
                                    if k in self.scenario.test_insts]))}}

        output_fn = [os.path.join(self.output_dir, 'cdf_train.png')]
        if self.train_test:
            output_fn.append(os.path.join(self.output_dir, 'cdf_test.png'))

        for inst_set, out in zip(['train', 'test'], output_fn):
            f = plt.figure(1, dpi=100, figsize=(10,10))
            ax1 = f.add_subplot(1,1,1)
            ax1.step(data['default'][inst_set][0],
                     data['default'][inst_set][1], color='red',
                     linestyle='-', label='default {}'.format(inst_set))
            ax1.step(data['incumbent'][inst_set][0],
                     data['incumbent'][inst_set][1], color='blue',
                     linestyle='-', label='incumbent {}'.format(inst_set))
            ax1.legend()
            ax1.grid(True)
            ax1.set_xscale('log')
            ax1.set_ylabel('probability of being solved')
            ax1.set_xlabel('time')
            # Plot 'timeout'
            if timeout:
                ax1.text(timeout,
                         ax1.get_ylim()[0] - 0.1 * np.abs(ax1.get_ylim()[0]),
                         "timeout ", horizontalalignment='center',
                         verticalalignment="top", rotation=30)
                ax1.axvline(x=timeout, linestyle='--')

            f.tight_layout()
            f.savefig(out)
            plt.close(f)
        if len(output_fn) == 1:
            return output_fn[0]
        return output_fn

    def configurator_footprint(self, scen, runhistories, incumbents=None,
                               max_confs_plot=1000, time_slider=False, num_quantiles=10):
        """
        Parameters
        ----------
        scen: Scenario
            scenario
        rhs: List[RunHistory]
            (unvalidated!) runhistories
        inc: List[Configuration]
            incumbents of all runs
        max_confs_plot: int
            # configurations to be plotted
        time_slider: bool
            whether or not to have a time_slider-widget on cfp-plot
            INCREASES FILE-SIZE DRAMATICALLY
        num_quantiles: int
            if time_slider is not off, defines the number of quantiles for the

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

        cfp = ConfiguratorFootprint(
                       scenario=scen,
                       runhistory=runhistories[0],
                       incs=incumbents, max_plot=max_confs_plot,
                       output_dir=self.output_dir,
                       time_slider=time_slider,
                       num_quantiles=num_quantiles)
        try:
            r = cfp.run()
        except MemoryError as err:
            self.logger.error(err)
            raise MemoryError("Memory Error occured in configurator footprint. "
                              "You may want to reduce the number of plotted "
                              "configs (using the '--cfp_max_plot'-argument)")
        return r

    def plot_parallel_coordinates(self, rh, output, params, n_configs, validator):
        """ Plot parallel coordinates (visualize higher dimensions), here used
        to visualize pcs. This function prepares the data from a SMAC-related
        format (using runhistories and parameters) to a more general format
        (using a dataframe). The resulting dataframe is passed to the
        parallel_coordinates-routine.

        NOTE: the given runhistory should contain only optimization and no
        validation to analyze the explored parameter-space.

        Parameters
        ----------
        rh: RunHistory
            rundata to take configs from
        output: str
            where to save plot
        params: list[str]
            parameters to be plotted
        n_configs: int
            max # configs
        validator: Validator
            to calculate alpha values

        Returns
        -------
        output: str
            path to plot
        """
        parallel_coordinates_plotter = ParallelCoordinatesPlotter(rh, self.output_dir,
                                                                  validator, self.scenario.cs,
                                                                  runtime=self.scenario.run_obj == 'runtime')
        output = parallel_coordinates_plotter.plot_n_configs(n_configs, params)
        return output

    def _get_mean_var_time(self, validator, traj, pred, rh):
        validator.traj = traj  # set trajectory
        time, configs = [], []

        if pred:
            for entry in traj:
                time.append(entry["wallclock_time"])
                configs.append(entry["incumbent"])
                #self.logger.debug('Time: %d Runs: %d', time[-1],
                #                  len(rh.get_runs_for_config(configs[-1])))

            self.logger.debug("Using %d samples (%d distinct) from trajectory.",
                              len(time), len(set(configs)))

            if validator.epm:  # not log as validator epm is trained on cost, not log cost
                epm = validator.epm
            else:
                self.logger.debug("No EPM passed! Training new one from runhistory.")
                # Train random forest and transform training data (from given rh)
                # Not using validator because we want to plot uncertainties
                rh2epm = RunHistory2EPM4Cost(num_params=len(self.scenario.cs.get_hyperparameters()),
                                             scenario=self.scenario)
                X, y = rh2epm.transform(rh)
                self.logger.debug("Training model with data of shape X: %s, y:%s",
                                  str(X.shape), str(y.shape))

                types, bounds = get_types(self.scenario.cs, self.scenario.feature_array)
                epm = RandomForestWithInstances(types=types,
                                                bounds=bounds,
                                                instance_features=self.scenario.feature_array,
                                                #seed=self.rng.randint(MAXINT),
                                                ratio_features=1.0)
                epm.train(X, y)
            config_array = convert_configurations_to_array(configs)
            mean, var = epm.predict_marginalized_over_instances(config_array)
            var = np.zeros(mean.shape)
            # We don't want to show the uncertainty of the model but uncertainty over multiple optimizer runs
            # This variance is computed in an outer loop.
        else:
            mean, var = [], []
            for entry in traj:
                time.append(entry["wallclock_time"])
                configs.append(entry["incumbent"])
                costs = _cost(configs[-1], rh, rh.get_runs_for_config(configs[-1]))
                #self.logger.debug(len(costs), time[-1]
                if not costs:
                    time.pop()
                else:
                    mean.append(np.mean(costs))
                    var.append(0)  # No variance over instances
            mean, var = np.array(mean).reshape(-1, 1), np.array(var).reshape(-1, 1)
        return mean, var, time

    def plot_cost_over_time(self, rh: RunHistory, runs: List[ConfiguratorRun],
                            output_fn: str="performance_over_time.png",
                            validator: Union[None, Validator]=None):
        """ Plot performance over time, using all trajectory entries
            with max_time = wallclock_limit or (if inf) the highest
            recorded time

            Parameters
            ----------
            rh: RunHistory
                runhistory to use
            runs: List[SMACrun]
                list of configurator-runs
            output_fn: str
                path to output-png
            validator: TODO description
        """
        self.logger.debug("Estimating costs over time for best run.")
        validated = True

        if len(runs) > 1:
            means, times = [], []
            all_times = []
            for run in runs:
                # Ignore variances as we plot variance over runs
                validated = validated and run.traj
                mean, _, time = self._get_mean_var_time(validator, run.traj, not
                                                        run.validated_runhistory, rh)
                means.append(mean.flatten())
                all_times.extend(time)
                times.append(time)
            means = np.array(means)
            times = np.array(times)
            all_times = np.array(sorted(all_times))
            at = [0 for _ in runs]  # keep track at which timestep each trajectory is
            m = [np.nan for _ in runs]  # used to compute the mean over the timesteps
            mean = np.ones((len(all_times), 1)) * -1
            var = np.ones((len(all_times), 1)) * -1
            upper = np.ones((len(all_times), 1)) * -1
            lower = np.ones((len(all_times), 1)) * -1
            for time_idx, t in enumerate(all_times):
                for traj_idx, entry_idx in enumerate(at):
                    try:
                        if t == times[traj_idx][entry_idx]:
                            m[traj_idx] = means[traj_idx][entry_idx]
                            at[traj_idx] += 1
                    except IndexError:
                        pass  # Reached the end of one trajectory. No need to check it further
                # var[time_idx][0] = np.nanvar(m)
                u, l, m_ = np.nanpercentile(m, 75), np.nanpercentile(m, 25), np.nanpercentile(m, 50)
                # self.logger.debug((mean[time_idx][0] + np.sqrt(var[time_idx][0]), mean[time_idx][0],
                #                   mean[time_idx][0] - np.sqrt(var[time_idx][0])))
                # self.logger.debug((l, m_, u))
                upper[time_idx][0] = u
                mean[time_idx][0] = m_
                lower[time_idx][0] = l
            time = all_times
        else:  # no new statistics computation necessary
            validated = True if runs[0].validated_runhistory else False
            mean, var, time = self._get_mean_var_time(validator, runs[0].traj, not validated, rh)
            upper = lower = mean

        mean = mean[:, 0]
        upper = upper[:, 0]
        lower = lower[:, 0]

        uncertainty_upper = upper  # mean + np.sqrt(var)
        uncertainty_lower = lower  # mean - np.sqrt(var)
        clip_y_lower = False
        if self.scenario.run_obj == 'runtime':  # y-axis on log -> clip plot
            # Determine clipping point from lowest legal value
            clip_y_lower = min(list(uncertainty_lower[uncertainty_lower > 0])
                               + list(mean)) * 0.8
            uncertainty_lower[uncertainty_lower <= 0] = clip_y_lower * 0.9

        time_double = [t for sub in zip(time, time) for t in sub][1:-1]
        mean_double = [t for sub in zip(mean, mean) for t in sub][:-2]
        source = ColumnDataSource(data=dict(
                    x=time_double,
                    y=mean_double,
                    epm_perf=mean_double))

        hover = HoverTool(tooltips=[
                ("performance", "@epm_perf"),
                ("at-time", "@x")])

        p = figure(plot_width=700, plot_height=500, tools=[hover],
                   x_range=Range1d(max(min(time), 1), max(time)),
                   x_axis_type='log',
                   y_axis_type='log' if self.scenario.run_obj=='runtime' else 'linear',
                   title="Cost over time")

        if clip_y_lower:
            p.y_range = Range1d(clip_y_lower, 1.2 * max(uncertainty_upper))

        # start after 1% of the configuration budget
        # p.x_range = Range1d(min(time) + (max(time) - min(time)) * 0.01, max(time))

        ## Plot
        label = self.scenario.run_obj
        label = '{}{}'.format('validated ' if validated else 'estimated ', label)
        p.line('x', 'y', source=source, legend=label)

        # Fill area (uncertainty)
        # Defined as sequence of coordinates, so for step-effect double and
        # arange accordingly ([(t0, v0), (t1, v0), (t1, v1), ... (tn, vn-1)])
        time_double = [t for sub in zip(time, time) for t in sub][1:-1]
        uncertainty_lower_double = [u for sub in zip(uncertainty_lower,
            uncertainty_lower) for u in sub][:-2]
        uncertainty_upper_double = [u for sub in zip(uncertainty_upper,
            uncertainty_upper) for u in sub][:-2]
        band_x = np.append(time_double, time_double[::-1])
        band_y = np.append(uncertainty_lower_double, uncertainty_upper_double[::-1])
        p.patch(band_x, band_y, color='#7570B3', fill_alpha=0.2)

        # Format labels as 10^x
        p.xaxis.major_label_orientation = 3/4
        p.xaxis.ticker = AdaptiveTicker(mantissas=[1], base=10)

        p.legend.location = "bottom_left"

        p.xaxis.axis_label = "time (sec)"
        p.yaxis.axis_label = label
        p.xaxis.axis_label_text_font_size = "15pt"
        p.yaxis.axis_label_text_font_size = "15pt"
        p.xaxis.major_label_text_font_size = "12pt"
        p.yaxis.major_label_text_font_size = "12pt"
        p.title.text_font_size = "15pt"
        p.legend.label_text_font_size = "15pt"

        script, div = components(p)

        export_bokeh(p, os.path.join(self.output_dir, output_fn), self.logger)

        return script, div
