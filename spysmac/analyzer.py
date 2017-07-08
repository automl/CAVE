import os
import sys
import logging as log

import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from smac.runhistory.runhistory import RunKey, RunValue

from spysmac.html.html_builder import HTMLBuilder
from spysmac.plot.plotter import Plotter

class Analyzer(object):
    """
    Analyze SMAC-output data.
    Compares two configurations (default vs incumbent).
    PAR10, timeouts, scatterplots, etc.
    """

    def __init__(self, scenario, runhistory, incumbent, output=None):
        self.logger = log.getLogger("analyzer")

        self.output = output
        if not os.path.exists(output):
            os.makedirs(output)

        self.scenario = scenario
        self.runhistory = runhistory
        self.incumbent = incumbent
        self.train_inst = self.scenario.train_insts
        self.test_inst = self.scenario.test_insts

    def analyze(self):
        """
        Performs analysis of scenario by scrutinizing the runhistory.
        """
        default = self.scenario.cs.get_default_configuration()
        incumbent = self.incumbent
        # Extract data from runhistory
        default_performance = self.get_performance_per_instance(default,
                aggregate=np.mean)
        incumbent_performance = self.get_performance_per_instance(incumbent,
                aggregate=np.mean)
        self.logger.debug("Length default-cost %d, length inc-cost %d",
                len(default_performance), len(incumbent_performance))

        # Plotting
        plotter = Plotter()
        # Scatterplot
        plotter.plot_scatter(default_performance, incumbent_performance,
                output=os.path.join(self.output, 'scatter.png'))
        # CDF
        plotter.plot_cdf(default_performance, "default",
                output=os.path.join(self.output, 'def_cdf.png'))
        plotter.plot_cdf(incumbent_performance, "incumbent",
                output=os.path.join(self.output, 'inc_cdf.png'))

    def build_html(self):
        builder = HTMLBuilder(self.output, "SpySMAC")
        scatter_path = os.path.join(self.output, 'scatter.png')
        cdf_default_path = os.path.join(self.output, 'def_cdf.png')
        cdf_incumbent_path = os.path.join(self.output, 'inc_cdf.png')

        website = {"Scatterplot": {
                        "tooltip": "Scatterplot default vs incumbent", #str|None,
                        #"subtop1": {  # generates a further bottom if it is dictionary
                        #        "tooltip": str|None,
                        #        ...
                        #        }
                        #"table": table, #str|None (html table)
                        "figure" : scatter_path # str | None (file name)
                        },
                   "Cumulative distribution function (CDF)": {
                       "tooltip": "CDF for incumbent and for default",
                       "subtop1": {"figure": cdf_default_path},
                       "subtop2": {"figure": cdf_incumbent_path},
                       }
                  }
        builder.generate_html(website)

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

