import sys
import os
import logging

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use(os.path.join(os.path.dirname(__file__), 'mpl_style'))
from matplotlib import ticker

from ConfigSpace.util import impute_inactive_values

from smac.utils.validate import Validator
from smac.configspace import Configuration, convert_configurations_to_array
from smac.epm.rf_with_instances import RandomForestWithInstances
from smac.optimizer.objective import average_cost
from smac.runhistory.runhistory2epm import RunHistory2EPM4Cost
from smac.utils.util_funcs import get_types
from smac.runhistory.runhistory import RunHistory

from cave.plot.scatter import plot_scatter_plot
from cave.plot.confs_viz.viz_sampled_confs import SampleViz
from cave.plot.parallel_coordinates import ParallelCoordinatesPlotter

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

    def __init__(self, scenario, train_test, conf1_runs, conf2_runs, output):
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
        output: str
            output-directory
        """
        self.logger = logging.getLogger("cave.plotter")
        self.scenario = scenario
        self.train_test = train_test
        self.output = output
        self.vizrh = None

        # Split data into train and test
        data = {"default" : {"combined" : [], "train" : [], "test" : []},
                "incumbent" : {"combined" : [], "train" : [], "test" : []}}
        train = scenario.train_insts
        test = scenario.test_insts
        # Create array for all instances
        for k in conf1_runs:
            data["default"]["combined"].append(conf1_runs[k])
            if k in train:
                data["default"]["train"].append(conf1_runs[k])
            if k in test:
                data["default"]["test"].append(conf1_runs[k])
        for k in conf2_runs:
            data["incumbent"]["combined"].append(conf2_runs[k])
            if k in train:
                data["incumbent"]["train"].append(conf2_runs[k])
            if k in test:
                data["incumbent"]["test"].append(conf2_runs[k])
        for c in ["default", "incumbent"]:
            for s in ["combined", "train", "test"]:
                data[c][s] = np.array(data[c][s])
        self.data = data

    def plot_scatter(self, output='scatter.png'):
        """
        Creates a scatterplot of the two configurations on the given set of
        instances.
        Saves plot to file.

        Parameters:
        -----------
        output: string
            path to save plot in
        """
        self.logger.debug("Plot scatter to %s", output)

        metric = self.scenario.run_obj
        timeout = self.scenario.cutoff
        labels = ["default cost", "incumbent cost"]

        if self.train_test:
            conf1 = (self.data["default"]["train"],
                    self.data["default"]["test"])
            conf2 = (self.data["incumbent"]["train"],
                    self.data["incumbent"]["test"])
        else:
            conf1 = (self.data["default"]["combined"],)
            conf2 = (self.data["incumbent"]["combined"],)

        fig = plot_scatter_plot(conf1, conf2,
                                labels, metric=metric,
                                user_fontsize=mpl.rcParams['font.size'],
                                max_val=timeout,
                                jitter_timeout=True)
        fig.savefig(output)
        plt.close(fig)

    def plot_cdf_compare(self, output="CDF_compare.png"):
        """
        Plot the cumulated distribution functions for given configurations,
        plots will share y-axis and if desired x-axis.
        Saves plot to file.

        Parameters
        ----------
        output: string
            filename, default: CDF_compare.png
        """
        self.logger.debug("Plot CDF to %s", output)

        timeout = self.scenario.cutoff

        data = self.data

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
        data = {config_name : {label : prepare_data(x_data) for label, x_data in
                               data[config_name].items()}
                for config_name in data}

        # Until here the code is usable for an arbitrary number of
        # configurations. Below, it is specified for plotting default vs
        # incumbent only.

        if self.train_test:
            f = plt.figure(1, dpi=100, figsize=(10,5))
            ax1 = f.add_subplot(1,2,1)
            ax2 = f.add_subplot(1,2,2)
            ax1.step(data['default']['train'][0],
                     data['default']['train'][1], color='red',
                     linestyle='-', label='default train')
            ax1.step(data['incumbent']['train'][0],
                     data['incumbent']['train'][1], color='blue',
                     linestyle='-', label='incumbent train')
            ax2.step(data['default']['test'][0],
                     data['default']['test'][1], color='red',
                     linestyle='-', label='default test')
            ax2.step(data['incumbent']['test'][0],
                     data['incumbent']['test'][1], color='blue',
                     linestyle='-', label='incumbent test')
            ax2.legend()
            ax2.grid(True)
            ax2.set_xscale('log')
            ax2.set_ylabel('probability of being solved')
            ax2.set_xlabel('time')
            # Plot 'timeout'
            if timeout:
                ax2.text(timeout,
                         ax2.get_ylim()[0] - 0.1 * np.abs(ax2.get_ylim()[0]),
                         "timeout ", horizontalalignment='center',
                         verticalalignment="top", rotation=30)
                ax2.axvline(x=timeout, linestyle='--')

        else:
            f = plt.figure(1, dpi=100, figsize=(10,10))
            ax1 = f.add_subplot(1,1,1)
            ax1.step(data['default']['combined'][0],
                     data['default']['combined'][1], color='red',
                     label='default all instances')
            ax1.step(data['incumbent']['combined'][0],
                     data['incumbent']['combined'][1], color='blue',
                     label='incumbent all instances')
            ax1.set_title('PAR10 - CDF')

        # Always set props for ax1
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
        f.savefig(output)
        plt.close(f)

    def visualize_configs(self, scen, runhistories, incumbents=None, max_confs_plot=1000):
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
        """

        sz = SampleViz(scenario=scen,
                       runhistories=runhistories,
                       incs=incumbents, max_plot=max_confs_plot,
                       output_dir=self.output)
        r = sz.run()
        self.vizrh = sz.relevant_rh
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
        parallel_coordinates_plotter = ParallelCoordinatesPlotter(rh, self.output,
                                                                  validator, self.scenario.cs,
                                                                  runtime=self.scenario.run_obj == 'runtime')
        output = parallel_coordinates_plotter.plot_n_configs(n_configs, params)
        return output


    def plot_cost_over_time(self, rh, traj, output="performance_over_time.png",
                            validator=None):
        """ Plot performance over time, using all trajectory entries
            with max_time = wallclock_limit or (if inf) the highest
            recorded time

            Parameters
            ----------
            rh: RunHistory
                runhistory to use
            traj: List
                trajectory to take times/incumbents from
            output: str
                path to output-png
            epm: RandomForestWithInstances
                emperical performance model (expecting trained on all runs)
        """
        self.logger.debug("Estimating costs over time for best run.")
        validator.traj = traj  # set trajectory
        time, configs = [], []

        for entry in traj:
            time.append(entry["wallclock_time"])
            configs.append(entry["incumbent"])

        self.logger.debug("Using %d samples (%d distinct) from trajectory.",
                          len(time), len(set(configs)))

        if validator.epm:
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

        ## not necessary right now since the EPM only knows the features
        ## of the training instances
        # use only training instances
        #=======================================================================
        # if self.scenario.feature_dict:
        #     feat_array = []
        #     for inst in self.scenario.train_insts:
        #         feat_array.append(self.scenario.feature_dict[inst])
        #     backup_features_epm = epm.instance_features
        #     epm.instance_features = np.array(feat_array)
        #=======================================================================

        # predict performance for all configurations in trajectory
        config_array = convert_configurations_to_array(configs)
        mean, var = epm.predict_marginalized_over_instances(config_array)

        #=======================================================================
        # # restore feature array in epm
        # if self.scenario.feature_dict:
        #     epm.instance_features = backup_features_epm
        #=======================================================================

        mean = mean[:,0]
        var = var[:,0]
        uncertainty_upper = mean+np.sqrt(var)
        uncertainty_lower = mean-np.sqrt(var)

        # plot
        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.set_ylabel('performance')
        ax.set_xlabel('time [sec]')
        ax.plot(time, mean, 'r-', label="estimated performance")
        ax.fill_between(time, uncertainty_upper, uncertainty_lower, alpha=0.8,
                label="standard deviation")
        ax.set_xscale("log", nonposx='clip')

        ax.set_ylim(min(mean)*0.8, max(mean)*1.2)
        # start after 1% of the configuration budget
        ax.set_xlim(min(time)+(max(time) - min(time))*0.01, max(time))

        ax.legend()
        plt.tight_layout()
        fig.savefig(output)
        plt.close(fig)
