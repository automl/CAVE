import os
import logging

import numpy as np
import matplotlib.pyplot as plt
from typing import Union, List
plt.style.use(os.path.join(os.path.dirname(__file__), 'mpl_style'))  # noqa

from smac.utils.validate import Validator
from smac.runhistory.runhistory import RunHistory

from cave.reader.configurator_run import ConfiguratorRun
from cave.plot.scatter import plot_scatter_plot
from cave.plot.configurator_footprint import ConfiguratorFootprint
from cave.plot.parallel_coordinates import ParallelCoordinatesPlotter
from cave.plot.cost_over_time import CostOverTime
from cave.utils.helpers import get_cost_dict_for_config


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
        # TODO docstring
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
            min_val = min(min_val, min(default_test), min(incumbent_test))

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
                if (timeout is not None) and (x_data[idx] >= timeout):
                    x_data[idx] = timeout
                    y_data[idx] = y_data[idx-1]
            return (x_data, y_data)

        # Generate y_data
        data = {'default':
                {'train': prepare_data(
                           np.array([v for k, v in
                                    get_cost_dict_for_config(runhistory, default).items()
                                    if k in self.scenario.train_insts])),
                 'test': prepare_data(
                           np.array([v for k, v in
                                    get_cost_dict_for_config(runhistory, default).items()
                                    if k in self.scenario.test_insts]))},
                'incumbent':
                {'train': prepare_data(
                           np.array([v for k, v in
                                    get_cost_dict_for_config(runhistory, incumbent).items()
                                    if k in self.scenario.train_insts])),
                 'test': prepare_data(
                           np.array([v for k, v in
                                    get_cost_dict_for_config(runhistory, incumbent).items()
                                    if k in self.scenario.test_insts]))}}

        output_fn = [os.path.join(self.output_dir, 'cdf_train.png')]
        if self.train_test:
            output_fn.append(os.path.join(self.output_dir, 'cdf_test.png'))

        for inst_set, out in zip(['train', 'test'], output_fn):
            f = plt.figure(1, dpi=100, figsize=(10, 10))
            ax1 = f.add_subplot(1, 1, 1)
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
        cost_over_time = CostOverTime()
        return cost_over_time.plot(rh, runs, output_fn, validator)
