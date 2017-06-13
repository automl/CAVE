import os
import sys
import logging as log

import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

# Adjust the PYTHON_PATH to find the submodules
spysmac_path = os.path.dirname(os.path.realpath(__file__))[:-8]
sys.path = [os.path.join(spysmac_path, "plotting_scripts"),
            os.path.join(spysmac_path, "smac"),
            os.path.join(spysmac_path, "asapy"),
            ] + sys.path

from smac.runhistory.runhistory import RunKey, RunValue

from plottingscripts.plotting.scatter import plot_scatter_plot

from asapy.out_builder.html_builder import HTMLBuilder

class Analyzer(object):
    """
    Analyze SMAC-output data.
    Compares two configurations (default vs incumbent).
    PAR10, timeouts, scatterplots, etc.
    """

    def __init__(self, scenario, runhistory, train_inst, test_inst=None,
                 output=None):
        self.logger = log.getLogger("analyzer")

        self.output = output if output else scenario.output_dir

        self.scenario = scenario
        self.runhistory = runhistory
        self.train_inst = train_inst
        self.test_inst = test_inst

    def build_html(self):
        builder = HTMLBuilder(self.output, "SpySMAC")
        builder.generate_html(None)

    def get_performance_per_instance(self, conf, aggregate=None):
        """
        Aggregates performance for configuration on given over seeds.
        Raises LookupError if instance not evaluated on configuration.

        Parameters:
        -----------
        conf: Configuration
            configuration to evaluate
        aggregate: Function or None
            used to aggreagate performance over different seeds, takes a list as
            argument

        Returns:
        --------
        performance: dict(instance->performance)
            performance per instance (aggregated or as list per seeds)
        """
        # Check if config is in runhistory
        conf_id = self.runhistory.config_ids[conf]

        # Map instances to seeds in dict
        runs = self.runhistory.get_runs_for_config(conf)
        instance_seeds = dict()
        for r in runs:
            i, s = r
            if i in instance_seeds:
                instance_seeds[i].append(s)
            else:
                instance_seeds[i] = [s]

        # Get cost per instance
        instance_costs = {i: [self.runhistory.data[RunKey(conf_id, i, s)].cost for s in
                              instance_seeds[i]] for i in instance_seeds}

        # Aggregate:
        if aggregate:
            instance_costs = {i: aggregate(instance_costs[i]) for i in instance_costs}

        return instance_costs

    def get_performances_per_instance(self, conf1, conf2, aggregate=None, keep_missing=False):
        """
        Get performances for two configurations

        Parameters
        ----------
        conf1, conf2: Configuration
            configs to be compared
        aggregate: Function or None
            used to aggreagate performance over different seeds, takes a list as
            argument
        keep_missing: Boolean
            keep instances with one or no configs for it

        Returns
        -------
        instance_performances: dict(instance->tuple(perf1, perf2))
            performances for both configurations
        """
        conf1_perf = self.get_performance_per_instance(conf1, aggregate=aggregate)
        conf2_perf = self.get_performance_per_instance(conf2, aggregate=aggregate)
        instance_perf = dict()
        for i in set(conf1_perf.keys()) | set(conf2_perf.keys()):
            instance_perf[i] = (conf1_perf.get(i), conf2_perf.get(i))

        if not keep_missing:
            before = len(instance_perf)
            instance_perf = {i: instance_perf[i] for i in instance_perf if
                             instance_perf[i][0] and instance_perf[i][1]}
            self.logger.info("Remove {}/{} instances because they are not "
                             "evaluated on both configurations.".format(before -
                                 len(instance_perf), before))
        return instance_perf

    def scatterplot(self, conf1, conf2, instances=None, labels=('default', 'incumbent'),
                    title='SpySMAC scatterplot', metric='runtime'):
        """
        Creates a scatterplot of the two configurations on the given set of
        instances.

        Parameters:
        -----------
        conf1, conf2: Configuration
            configurations to compare
        instances: list of strings
            list of instances to compare. if None, compare all that are
            evaluated on both configs
        label: tuple of strings
            labels of plot
        title: string
            title of plot
        metric: string
            metric, runtime or quality
        """
        data = self.get_performances_per_instance(conf1, conf2,
                keep_missing=False, aggregate=np.mean)
        if instances:
            data = {k: data[k] for k in data if k in instances}
        # Transpose performance-data to make two lists
        x_data, y_data = zip(*data.values())
        x_data, y_data = np.array(x_data), np.array(y_data)
        fig = plot_scatter_plot(x_data, y_data, labels, title=title,
                                metric=metric)
        fig.savefig(os.path.join(self.output, "scatter.png"))

    def plot_cdf(self, conf, filename="CDF"):
        """
        Plot the cumulated distribution function for given configuration

        Parameters
        ----------
        conf: Configuration
            configuration to be plotted
        filename: string
            filename (without extension)
        """
        # TODO make PDF/CDF statistically robust
        # TODO encapsulate
        import matplotlib.pyplot as plt
        # Get dict and turn into sorted list
        data = self.get_performance_per_instance(conf, aggregate=np.mean)
        data = sorted(list(data.values()))
        y_data = np.array(range(len(data)))/(len(data)-1)
        # Plot1
        plt.plot(data, y_data)
        plt.ylabel('Probability of being solved')
        plt.xlabel('Time')
        plt.title('SpySMAC CDF')
        plt.grid(True)
        plt.savefig(os.path.join(self.output, filename + ".png"))
        plt.close()

