import sys
import os

import numpy as np
import matplotlib.pyplot as plt

from plottingscripts.plotting.scatter import plot_scatter_plot


class Plotter(object):
    """
    Responsible for plotting (scatter, CDF, etc.)
    """

    def plot_scatter(self, perf_conf1, perf_conf2, labels=('default', 'incumbent'),
                     title='SpySMAC scatterplot', metric='runtime',
                     output='scatter.png'):
        """
        Creates a scatterplot of the two configurations on the given set of
        instances.

        Parameters:
        -----------
        perf_conf1, perf_conf2: dict(string->float)
            dicts with instance->cost mapping
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

        # Only consider instances that are present in both configs
        for i in perf_conf1:
            if i in perf_conf2:
                conf1.append(perf_conf1[i])
                conf2.append(perf_conf2[i])

        ### Zero-tolerance: (TODO activate as soon as validation works)
        #for i in perf_conf1:
        #    conf1.append(perf_conf1[i])
        #    conf2.append(perf_conf2.pop(i))
        #if not len(perf_conf2) == 0:
        #    raise ValueError("Bad input to scatterplot.")

        fig = plot_scatter_plot(np.array(conf1), np.array(conf2),
                                labels, title=title, metric=metric)
        fig.savefig(output)
        fig.close()

    def plot_cdf(self, perf_conf, config_name, output="CDF.png"):
        """
        Plot the cumulated distribution function for given configuration

        Parameters
        ----------
        perf_conf: dict(string->float)
            instance-cost dict
        output: string
            filename
        """
        # TODO encapsulate
        # Get dict and turn into sorted list
        data = perf_conf
        data = sorted(list(data.values()))
        y_data = np.array(range(len(data)))/(len(data)-1)
        # Plot1
        plt.plot(data, y_data)
        plt.ylabel('Probability of being solved')
        plt.xlabel('Time')
        plt.title('{} - SpySMAC CDF'.format(config_name))
        plt.grid(True)
        plt.savefig(output)
        plt.close()
