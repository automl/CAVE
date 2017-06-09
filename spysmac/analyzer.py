import os
import sys

import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

# Adjust the PYTHON_PATH to find the submodules
spysmac_path = os.path.dirname(os.path.realpath(__file__))[:-8]
sys.path = [os.path.join(spysmac_path, "plotting_scripts"),
            os.path.join(spysmac_path, "smac"),
            ] + sys.path

from smac.runhistory.runhistory import RunKey, RunValue

from plottingscripts.plotting.scatter import plot_scatter_plot


class Analyzer(object):
    """
    Analyze SMAC-output data.
    Compares two configurations (default vs incumbent).
    PAR10, timeouts, scatterplots, etc.
    """

    def __init__(self, scenario, runhistory, train_inst, test_inst=None,
                 output=None):
        self.output = output if output else scenario.output_dir

        self.scenario = scenario
        self.runhistory = runhistory
        self.train_inst = train_inst
        self.test_inst = test_inst

    def get_performance(self, conf, instances):
        """
        Aggregates performance for configuration on given set of instances.

        Parameters:
        -----------
        conf: Configuration
            configuration to evaluate
        instances: list of strings
            instances to be evaluated

        Returns:
        --------
        performance: np.vector(len(instances))
            performance per instance
        """
        # Check if config is in runhistory
        conf_id = self.runhistory.config_ids[conf]

        # TODO outsource as function
        # Map instances to seeds in dict
        runs = self.runhistory.get_runs_for_config(conf)
        instance_seeds = {i: [] for i in instances}
        for r in runs:
            instance_seeds[r[0]].append(r[1])

        # Assert that all instances have been run
        for i in instances:
            if not i in instance_seeds:
                raise LookupError("Instance {} not run for config!".format(i))

        # Get performances per instance
        performances = []
        for i in instances:
            runs_for_i = [self.runhistory.data[RunKey(conf_id, i, s)].cost for s in
                          instance_seeds[i]]
            # TODO simple average for proof of concept
            performances.append(sum(runs_for_i) / len(runs_for_i))
        return np.array(performances)

    def scatterplot(self, conf1, conf2, instances, labels=('default', 'incumbent'),
            title='SpySMAC scatterplot', metric='runtime'):
        """
        Creates a scatterplot of the two configurations on the given set of
        instances.

        Parameters:
        -----------
        conf1, conf2: Configuration
            configurations to compare
        instances: list of strings
            list of instances to compare
        label: tuple of strings
            labels of plot
        title: string
            title of plot
        metric: string
            metric, runtime or quality
        """
        x_data = self.get_performance(conf1, instances)
        y_data = self.get_performance(conf2, instances)
        fig = plot_scatter_plot(x_data, y_data, labels, title=title,
                                metric=metric)
        pp = PdfPages(os.path.join(self.output, 'scatter.pdf'))
        pp.savefig()
        pp.close()
        return fig
