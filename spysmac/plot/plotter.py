import sys
import os
import logging

import numpy as np
import matplotlib.pyplot as plt

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

    def __init__(self):
        self.logger = logging.getLogger("spysmac.plotter")

    def plot_scatter(self, conf_cost1, conf_cost2, timeout,
                     labels=('default cost', 'incumbent cost'),
                     metric='runtime',
                     output='scatter.png'):
        """
        Creates a scatterplot of the two configurations on the given set of
        instances.
        Saves plot to file.

        Parameters:
        -----------
        conf_cost1, conf_cost2: list(string,float)
            lists with instance,cost -tuples
        timeout: float
            timeout/cutoff
        label: tuple of strings
            labels of plot
        title: string
            title of plot
        metric: string
            metric, runtime or quality
        output: string
            path to save plot in
        """
        # Create data, raise error if instance not evaluated on one of the
        # configurations
        conf1, conf2 = [], []

        # Make sure both dicts have same keys
        if not set(conf_cost1) == set(conf_cost2):
            self.logger.warning("Bad input to scatterplot (unequal cost-dictionaries). "
                                "Will use only instances evaluated for both configs, "
                                "this might lead to unexpected behaviour or results.")

        # Only consider instances that are present in both configs
        for i in conf_cost1:
            if i in conf_cost2:
                conf1.append(conf_cost1[i])
                conf2.append(conf_cost2[i])

        if not timeout:
            timeout = max(max(conf1),max(conf2))

        conf1, conf2 = np.array(conf1), np.array(conf2)
        fig = plot_scatter_plot(conf1, conf2,
                                labels, metric=metric,
                                user_fontsize=12, max_val=timeout,
                                jitter_timeout=True)
        fig.savefig(output)
        plt.close(fig)

    def plot_cdf_compare(self, conf1_runs, conf2_runs,
                         timeout, run_obj, train=[], test=[],
                         output="CDF_compare.png"):
        """
        Plot the cumulated distribution functions for given configurations,
        plots will share y-axis and if desired x-axis.
        Saves plot to file.

        Parameters
        ----------
        conf1_runs, conf2_runs: list(RunValue)
            lists with smac.runhistory.runhistory.RunValue, from which to read
            cost or time
        timeout: float
            timeout/cutoff
        run_obj: str
            quality or runtime
        train: list(strings)
            train-instances, will be printed separately if test also specified
        test: list(strings)
            test-instances, will be printed separately if train also specified
        output: string
            filename, default: CDF_compare.png
        """
        # Make sure train and test is only differentiated if data is available
        if train == [[None]] or train == [None]:
            train = []
        if test == [[None]] or test == [None]:
            test = []

        def extract_cost(runvalue):
            """ extract either cost or time, depending on run_obj """
            if run_obj == "runtime":
                return runvalue.time
            else:
                return runvalue.cost

        # Split data into train and test
        data = {"default" : {"combined" : [], "train" : [], "test" : []},
                "incumbent" : {"combined" : [], "train" : [], "test" : []}}
        for k in conf1_runs:
            data["default"]["combined"].append(extract_cost(conf1_runs[k]))
            if k.instance in train:
                data["default"]["train"].append(extract_cost(conf1_runs[k]))
            if k.instance in test:
                data["default"]["test"].append(extract_cost(conf1_runs[k]))
        for k in conf2_runs:
            data["incumbent"]["combined"].append(extract_cost(conf2_runs[k]))
            if k.instance in train:
                data["incumbent"]["train"].append(extract_cost(conf2_runs[k]))
            if k.instance in test:
                data["incumbent"]["test"].append(extract_cost(conf2_runs[k]))

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

        if train and test:
            f, (ax1, ax2) = plt.subplots(1, 2)
            ax1.step(data['default']['train'][0],
                     data['default']['train'][1], color='red',
                     linestyle='--', label='default train')
            ax1.step(data['incumbent']['train'][0],
                     data['incumbent']['train'][1], color='blue',
                     linestyle='--', label='incumbent train')
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

    def visualize_configs(self, scen, rh):
        sz = SampleViz(scenario=scen,
                       runhistory=rh,
                       incs=None)
        sz.run()

