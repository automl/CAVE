import sys
import os
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker

from spysmac.plot.scatter import plot_scatter_plot
from spysmac.plot.confs_viz.viz_sampled_confs import SampleViz

__author__ = "Joshua Marben"
__copyright__ = "Copyright 2017, ML4AAD"
__license__ = "3-clause BSD"
__maintainer__ = "Joshua Marben"
__email__ = "joshua.marben@neptun.uni-freiburg.de"

class Plotter(object):
    """
    Responsible for plotting (scatter, CDF, etc.)
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
            f, (ax1, ax2) = plt.subplots(1, 2)
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

            ax1.set_title('Training - SpySMAC CDF')
            ax2.set_title('Test - SpySMAC CDF')
        else:
            f, ax1 = plt.subplots()
            ax1.step(data['default']['combined'][0],
                     data['default']['combined'][1], color='red',
                     label='default all instances')
            ax1.step(data['incumbent']['combined'][0],
                     data['incumbent']['combined'][1], color='blue',
                     label='incumbent all instances')
            ax1.set_title('PAR10 - SpySMAC CDF')

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

    def plot_parallel_coordinates(self, rh, output, params=None):
        """ Plotting a parallel coordinates plot, visualizing the explored PCS.
        """
        if not params:
            params = rh.get_all_configs()[0].keys()[:5]  # which parameters to plot
        full_index = params + ["cost", "runs"]  # indices for dataframe
        index = params  # plot only those
        x = [i for i, _ in enumerate(index)]

        def colour(category):
            """ TODO Returning colour for plotting, possibly dependent on
            category/cost?
            """
            # TODO
            return np.random.choice(['#2e8ad8', '#cd3785', '#c64c00', '#889a00'])

        # Create dataframe with configs + runs/cost
        data = []
        for conf in rh.get_all_configs():
            new_entry = {"cost":rh.get_cost(conf),
                         "runs":len(rh.get_runs_for_config(conf))}
            pa_d = conf.get_dictionary()
            for p in params:
                if isinstance(pa_d[p], str):
                    try:
                        new_entry[p] = int(pa_d[p])
                    except ValueError:
                        new_entry[p] = float(pa_d[p])
                else:
                    new_entry[p] = pa_d[p]
            data.append(pd.Series(new_entry))
        full_data = pd.DataFrame(data)
        data = full_data.drop(['cost', 'runs'], axis=1)

        # Create subplots
        fig, axes = plt.subplots(1, len(index)-1, sharey=False, figsize=(15,5))

        # Normalize the data for each parameter, so the displayed ranges are
        # meaningful.
        min_max = {}
        for p in index:
            min_max[p] = [data[p].min(), data[p].max(), np.ptp(data[p])]
            data[p] = np.true_divide(data[p] - data[p].min(), np.ptp(data[p]))

        # Plot data
        for i, ax in enumerate(axes):
            for idx in data.index:
                category = full_data.loc[idx, 'cost']
                ax.plot(x, data.loc[idx, index], colour(category))
            ax.set_xlim([x[i], x[i+1]])

        # Labeling axes
        num_ticks = 10
        for p, ax in enumerate(axes):
            ax.xaxis.set_major_locator(ticker.FixedLocator([p]))
            if p == len(axes)-1:
                # Move the final axis' ticks to the right-hand side
                ax = plt.twinx(axes[-1])
                p = len(axes)
                ax.xaxis.set_major_locator(ticker.FixedLocator([x[-2], x[-1]]))
            minimum, maximum, param_range = min_max[index[p]]
            step = param_range / float(num_ticks)
            # TODO adjust tick-labels to int/float/categorical and maybe even log?
            tick_labels = [round(minimum + step * i, 2) for i in
                    range(num_ticks+1)]
            norm_min = data[index[p]].min()
            norm_range = np.ptp(data[index[p]])
            norm_step = norm_range / float(num_ticks)
            ticks = [round(norm_min + norm_step * i, 2) for i in
                    range(num_ticks+1)]
            ax.yaxis.set_ticks(ticks)
            ax.set_yticklabels(tick_labels)
            if not p == len(axes)-1:
                ax.set_xticklabels([index[p]], rotation=5)
        ax.set_xticklabels([index[-2], index[-1]], rotation=5)

        # Remove space between subplots
        plt.subplots_adjust(wspace=0)

        plt.title("Explored parameter ranges in parallel coordinates.")

        fig.savefig(output)
        plt.close(fig)
