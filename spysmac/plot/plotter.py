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
                     labels=('default', 'incumbent'),
                     title='SpySMAC scatterplot', metric='runtime',
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
                                labels, title=title, metric=metric,
                                user_fontsize=10, max_val=timeout,
                                jitter_timeout=True)
        fig.savefig(output)
        plt.close(fig)

    def plot_cdf_compare(self, cost_conf1, config_name1, cost_conf2,
                         config_name2, timeout, same_x=True, output="CDF_compare.png"):
        """
        Plot the cumulated distribution functions for given configurations,
        plots will share y-axis and if desired x-axis.
        Saves plot to file.

        Parameters
        ----------
        cost_conf1, cost_conf2: dict(string->float)
            instance-cost dict
        config_name1, config_name2: str
            names to be printed on axis of plot
        timeout: float
            timeout/cutoff
        same_x: bool
            whether the two configs should share a single x-axis and plot
        output: string
            filename, default: CDF_compare.png
        """
        cost1, cost2 = sorted(list(cost_conf1.values())), sorted(list(cost_conf2.values()))
        y_cost1 = np.array(range(len(cost1)))/(len(cost1)-1)
        y_cost2 = np.array(range(len(cost2)))/(len(cost2)-1)
        # Manipulate data for timeouts
        for idx in range(len(cost1)):
            if cost1[idx] >= timeout:
                cost1[idx] = timeout
                y_cost1[idx] = y_cost1[idx-1]
        for idx in range(len(cost2)):
            if cost2[idx] >= timeout:
                cost2[idx] = timeout
                y_cost2[idx] = y_cost2[idx-1]


        if same_x:
            f, ax1 = plt.subplots()
            ax1.plot(cost1, y_cost1, color='red', label='config_name1')
            ax1.plot(cost2, y_cost2, color='blue', label='config_name2')
            ax1.set_title('{}+{} - SpySMAC CDF'.format(config_name1, config_name2))
        else:
            f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
            ax1.plot(cost1, y_cost1, color='red', label='config_name1')
            ax2.plot(cost2, y_cost2, color='blue', label='config_name2')
            ax1.set_title('{} - SpySMAC CDF'.format(config_name1))
            ax2.set_title('{} - SpySMAC CDF'.format(config_name2))

        # Always set props for ax1
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

