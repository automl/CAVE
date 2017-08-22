import sys
import os
import logging

import numpy as np
import matplotlib.pyplot as plt

from plottingscripts.plotting.scatter import plot_scatter_plot

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

    def plot_scatter(self, cost_conf1, cost_conf2, timeout,
                     labels=('default cost', 'incumbent cost'),
                     metric='runtime',
                     output='scatter.png'):
        """
        Creates a scatterplot of the two configurations on the given set of
        instances.
        Saves plot to file.

        Parameters:
        -----------
        cost_conf1, cost_conf2: dict(string->float)
            dicts with instance->cost mapping
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
        if not set(cost_conf1) == set(cost_conf2):
            self.logger.warning("Bad input to scatterplot (unequal cost-dictionaries). "
                                "Will use only instances evaluated for both configs, "
                                "this might lead to unexpected behaviour or results.")

        # Only consider instances that are present in both configs
        for i in cost_conf1:
            if i in cost_conf2:
                conf1.append(cost_conf1[i])
                conf2.append(cost_conf2[i])

        fig = plot_scatter_plot(np.array(conf1), np.array(conf2),
                                labels, metric=metric,
                                user_fontsize=12, max_val=timeout,
                                jitter_timeout=True)
        fig.savefig(output)
        plt.close(fig)

    def plot_cdf_compare(self, data, timeout, same_x=True,
                         train=[], test=[],
                         output="CDF_compare.png"):
        """
        Plot the cumulated distribution functions for given configurations,
        plots will share y-axis and if desired x-axis.
        Saves plot to file.

        Parameters
        ----------
        data: dict(string->dict(string->float))
            maps config-names to their instance-cost dicts
        timeout: float
            timeout/cutoff
        same_x: bool
            whether the two configs should share a single x-axis and plot
        train: list(strings)
            train-instances, will be printed separately if test also specified
        test: list(strings)
            test-instances, will be printed separately if train also specified
        output: string
            filename, default: CDF_compare.png
        """
        data = {config_name : {'combined' : sorted(list(data[config_name].values())),
                               'train' : sorted([c for i, c in
                                    data[config_name].items() if i in train]),
                               'test' : sorted([c for i, c in
                                    data[config_name].items() if i in test])}
                for config_name in data}

        def prepare_data(x_data):
            """ Helper function to keep things easy, generates y_data and
            manages x_data-timeouts """
            y_data = np.array(range(len(x_data)))/(len(x_data)-1)
            for idx in range(len(x_data)):
                if x_data[idx] >= timeout:
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

        if same_x:
            f, ax1 = plt.subplots()
            ax1.step(data['default']['combined'][0],
                     data['default']['combined'][1], color='red',
                     label='default allinst')
            ax1.step(data['incumbent']['combined'][0],
                     data['incumbent']['combined'][1], color='blue',
                     label='incumbent allinst')
            if train and test:
                ax1.step(data['default']['train'][0],
                         data['default']['train'][1], color='red',
                         linestyle='--', label='default train')
                ax1.step(data['incumbent']['train'][0],
                         data['incumbent']['train'][1], color='blue',
                         linestyle='--', label='incumbent train')
                ax1.step(data['default']['test'][0],
                         data['default']['test'][1], color='red',
                         linestyle='-.', label='default train')
                ax1.step(data['incumbent']['test'][0],
                         data['incumbent']['test'][1], color='blue',
                         linestyle='-.', label='incumbent test')

            ax1.set_title('{}+{} - SpySMAC CDF'.format('default', 'incumbent'))
        else:
            f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
            ax1.step(data['default']['combined'][0],
                     data['default']['combined'][1], color='red',
                     label='default allinst')
            ax2.step(data['incumbent']['combined'][0],
                     data['incumbent']['combined'][1], color='blue',
                     label='incumbent allinst')
            if train and test:
                ax1.step(data['default']['train'][0],
                         data['default']['train'][1], color='red',
                         linestyle='--', label='default train')
                ax2.step(data['incumbent']['train'][0],
                         data['incumbent']['train'][1], color='blue',
                         linestyle='--', label='incumbent train')
                ax1.step(data['default']['test'][0],
                         data['default']['test'][1], color='red',
                         linestyle='-.', label='default train')
                ax2.step(data['incumbent']['test'][0],
                         data['incumbent']['test'][1], color='blue',
                         linestyle='-.', label='incumbent test')
            ax1.set_title('{} - SpySMAC CDF'.format('default'))
            ax2.set_title('{} - SpySMAC CDF'.format('incumbent'))
            ax2.legend()

        # Always set props for ax1
        ax1.legend()
        ax1.grid(True)
        ax1.set_xscale('log')
        ax1.set_ylabel('Probability of being solved')
        ax1.set_xlabel('Time')
        # Plot 'timeout'
        ax1.text(timeout,
                 ax1.get_ylim()[0] - 0.1 * np.abs(ax1.get_ylim()[0]),
                 "timeout ",  horizontalalignment='center',
                 verticalalignment="top", rotation=30)
        ax1.axvline(x=timeout, linestyle='--')

        # Set props for ax2 if exists
        if not same_x:
            ax2.grid(True)
            ax2.set_xscale('log')
            ax2.set_xlabel('Time')
            ax2.text(timeout,
                     ax2.get_ylim()[0] - 0.1 * np.abs(ax2.get_ylim()[0]),
                     "timeout ",  horizontalalignment='center',
                     verticalalignment="top", rotation=30)
            ax2.axvline(x=timeout, linestyle='--')

        f.savefig(output)
        plt.close(f)

