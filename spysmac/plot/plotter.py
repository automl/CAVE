import sys
import os
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker

from ConfigSpace.util import impute_inactive_values

from smac.utils.validate import Validator

from spysmac.plot.scatter import plot_scatter_plot
from spysmac.plot.confs_viz.viz_sampled_confs import SampleViz
from spysmac.plot.parallel_coordinates import plot_parallel_coordinates

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

    def __init__(self, scenario, train_test, conf1_runs, conf2_runs):
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
        """
        self.logger = logging.getLogger("spysmac.plotter")
        self.scenario = scenario
        self.train_test = train_test

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
                                user_fontsize=12, max_val=timeout,
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
            ax2.set_ylabel('Probability of being solved')
            ax2.set_xlabel('Time')
            # Plot 'timeout'
            if timeout:
                ax2.text(timeout,
                         ax2.get_ylim()[0] - 0.1 * np.abs(ax2.get_ylim()[0]),
                         "timeout ", horizontalalignment='center',
                         verticalalignment="top", rotation=30)
                ax2.axvline(x=timeout, linestyle='--')

            ax1.set_title('Training - CDF')
            ax2.set_title('Test - CDF')
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
        ax1.set_ylabel('Probability of being solved')
        ax1.set_xlabel('Time')
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

    def visualize_configs(self, scen, rh, inc=None):
        sz = SampleViz(scenario=scen,
                       runhistory=rh,
                       incs=inc)
        return sz.run()

    def plot_parallel_coordinates(self, rh, output, params):
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

        Returns
        -------
        output: str
            path to plot
        """
        # TODO: plot only good configurations/configurations with at least n runs

        # Get ALL parameter names and metrics to be passed on
        parameter_names = impute_inactive_values(rh.get_all_configs()[0]).keys()
        metrics = ["cost", "runs"]

        full_index = params + metrics  # indices for dataframe

        # Create dataframe with configs + runs/cost
        data = []
        for conf in rh.get_all_configs():
            # Add metrics
            new_entry = {"cost":rh.get_cost(conf),
                         "runs":len(rh.get_runs_for_config(conf))}
            # Complete configuration with imputed unused values
            conf_dict = impute_inactive_values(conf).get_dictionary()
            for p in params:
                # TODO handle log-scales and unused parameters
                # No strings allowed for plotting -> cast to numerical
                if isinstance(conf_dict[p], str):
                    # Catch "on" and "off"
                    if conf_dict[p] == "on":
                        new_entry[p] = 1
                    elif conf_dict[p] == "off":
                        new_entry[p] = 0
                    try:
                        new_entry[p] = int(conf_dict[p])
                    except ValueError:
                        new_entry[p] = float(conf_dict[p])
                else:
                    new_entry[p] = conf_dict[p]
            data.append(pd.Series(new_entry))
        full_data = pd.DataFrame(data)

        plot_parallel_coordinates(full_data, params, output)
        return output


    def plot_cost_over_time(self, rh, traj, output="performance_over_time.png"):
        """ Plot performance over time according to SMACs validate-function. """
        self.logger.info("Estimating costs over time for best run.")
        validator = Validator(self.scenario, trajectory=traj, output="")
        time, configs, traj_costs = [], [], []
        if (np.isfinite(self.scenario.wallclock_limit)):
            max_time = self.scenario.wallclock_limit
        else:
            max_time = traj[-1][mode]
        counter = 2**0
        for entry in traj[::-1]:
            if (entry["wallclock_time"] <= max_time/counter):
                time.append(entry["wallclock_time"])
                configs.append(entry["incumbent"])
                traj_costs.append(entry["cost"])
                counter *= 2
        if not traj[0]["incumbent"] in configs:
            configs.append(traj[0]["incumbent"])
            traj_costs.append(traj[0]["cost"])
            time.append(traj[0]["wallclock_time"])  # add first

        validated_rh = validator.validate_epm(list(set(configs)), 'train+test', 1,
                                              runhistory=rh)
        # Plot performances over time
        costs = [validated_rh.get_cost(c) for c in configs]

        self.logger.debug(time)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        # Plot in reverse order!
        # TODO train/test
        time, costs, traj_costs = time[::-1], costs[::-1], traj_costs[::-1]
        ax.plot(time, costs, 'r-', time, label="Actual Costs")
        ax.plot(traj_costs, 'b--', label="Costs of trajectory")
        ax.set_xscale("log", nonposx='clip')
        # Set y-limits in case that traj_costs are very high and ruin the plot
        ax.set_ylim(min(min(costs), min(traj_costs)), max(costs)+max(costs)*0.1)
        ax.legend()

        plt.title("Performance over time.")

        fig.savefig(output)
        plt.close(fig)
