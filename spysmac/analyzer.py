import os
import sys
import logging as log
from collections import OrderedDict

import numpy as np
from pandas import DataFrame

from smac.optimizer.objective import average_cost
from smac.runhistory.runhistory import RunKey, RunValue, RunHistory
from smac.utils.io.traj_logging import TrajLogger
from smac.utils.io.input_reader import InputReader
from smac.scenario.scenario import Scenario
from smac.utils.validate import Validator

from pimp.importance.importance import Importance

from spysmac.html.html_builder import HTMLBuilder
from spysmac.plot.plotter import Plotter


__author__ = "Joshua Marben"
__copyright__ = "Copyright 2017, ML4AAD"
__license__ = "3-clause BSD"
__maintainer__ = "Joshua Marben"
__email__ = "joshua.marben@neptun.uni-freiburg.de"

class Analyzer(object):
    """
    Analyze SMAC-output data.
    Compares two configurations (default vs incumbent).
    PAR10, timeouts, scatterplots, etc.
    """

    def __init__(self, scenario_fn, runhistory_fn, traj_fn, output):
        """
        Constructor

        Arguments
        ---------
        scenario_fn: string
            scenario file for train/test and cutoff
        runhistory_fn: string
            runhistory file from which to take data
        traj_fn: string
            trajectory file
        output: string
            output to which to write
        """
        self.logger = log.getLogger("spysmac.analyzer")

        # Create output if necessary
        self.output = output
        self.logger.info("Writing to %s", self.output)
        if not os.path.exists(output):
            os.makedirs(output)

        # Create Scenario (disable output_dir to avoid cluttering)
        in_reader = InputReader()
        scen = in_reader.read_scenario_file(scenario_fn)
        scen['output_dir'] = ""
        self.scenario = Scenario(scen)

        # Load runhistory and trajectory
        self.runhistory = RunHistory(average_cost)
        self.runhistory.load_json(runhistory_fn, self.scenario.cs)
        self.trajectory = TrajLogger.read_traj_aclib_format(fn=traj_fn,
                cs=self.scenario.cs)

        self.incumbent = self.trajectory[-1]['incumbent']
        self.train_inst = self.scenario.train_insts
        self.test_inst = self.scenario.test_insts

        # Generate missing data via validation
        new_rh_path = os.path.join(output, 'validated_rh.json')
        self.logger.info("Validating to complete data, saving validated "
                    "runhistory in %s.", new_rh_path)
        validator = Validator(self.scenario, self.trajectory, new_rh_path) # args_.seed)
        self.runhistory = validator.validate('def+inc', 'train+test', 1, -1,
                self.runhistory, None)

        self.importance = Importance(scenario_fn, runhistory_fn,
                                parameters_to_evaluate=-1,
                                traj_file=traj_fn,
                                save_folder=os.path.join(output, "PIMP"))  # create importance object

        # Paths
        self.scatter_path = os.path.join(self.output, 'scatter.png')
        self.cdf_combined_path = os.path.join(self.output, 'def_inc_cdf_comb.png')
        self.cdf_single_path = os.path.join(self.output, 'def_inc_cdf_single.png')

    def analyze(self):
        """
        Performs analysis of scenario by scrutinizing the runhistory.
        Calculate PAR10-values and plot CDF and scatter.
        """
        default = self.scenario.cs.get_default_configuration()
        incumbent = self.incumbent
        # Extract data from runhistory
        default_cost = self.get_cost_per_instance(default, aggregate=np.mean)
        incumbent_cost = self.get_cost_per_instance(incumbent, aggregate=np.mean)
        if not len(default_cost) == len(incumbent_cost):
            self.logger.warning("Default evaluated on %d instances, "
                                "incumbent evaluated on %d instances! "
                                "Might lead to unexpected results, consider "
                                "re-validating your results.",
                                len(default_cost),
                                len(incumbent_cost))

        self.overview = self.create_overview_table()

        # Analysis
        self.logger.debug("Calculate par10-values")
        self.calculate_par10(default_cost, incumbent_cost)
        self.par10_table = self.create_html_table()

        # Plotting
        plotter = Plotter()
        # Scatterplot
        self.logger.debug("Plot scatter")
        plotter.plot_scatter(default_cost, incumbent_cost,
                             output=self.scatter_path,
                             timeout=self.scenario.cutoff)
        # CDF
        self.logger.debug("Plot CDF")
        cost_dict = {'default' : default_cost, 'incumbent' : incumbent_cost}
        plotter.plot_cdf_compare(cost_dict,
                                 timeout= self.scenario.cutoff,
                                 same_x=True,
                                 #train=self.train_inst, test=self.test_inst,
                                 output=self.cdf_combined_path)
        plotter.plot_cdf_compare(cost_dict,
                                 timeout= self.scenario.cutoff,
                                 same_x=False,
                                 #train=self.train_inst, test=self.test_inst,
                                 output=self.cdf_single_path)

        self.importance.evaluate_scenario(evaluation_method='all')

    def build_html(self):
        """ Build website using the HTMLBuilder. Return website as dictionary
        for further stacking. Also saves website in
        'self.output/SpySMAC/report.html'

        Return
        ------
        website: dict
            website in dict as used in HTMLBuilder, can be stacked recursively
            into another dict
        """
        builder = HTMLBuilder(self.output, "SpySMAC")

        website = OrderedDict([
                   ("Overview",
                    {"table": self.overview}),
                   ("PAR10",
                    {"table": self.par10_table}),
                   ("Scatterplot",
                    {"figure" : self.scatter_path}),
                   ("Cumulative distribution function (CDF)",
                    {
                     "Single": {"figure": self.cdf_single_path},
                     "Combined": {"figure": self.cdf_combined_path},
                    }),
                   #("Parameter Importance" :
                       
                  ])
        builder.generate_html(website)
        return website

    def get_cost_per_instance(self, conf, aggregate=None):
        """
        Aggregates cost for configuration on given over seeds.
        Raises LookupError if instance not evaluated on configuration.

        Parameters:
        -----------
        conf: Configuration
            configuration to evaluate
        aggregate: function or None
            used to aggregate cost over different seeds, takes a list as
            argument

        Returns:
        --------
        cost: dict(instance->cost)
            cost per instance (aggregated or as list per seeds)
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

    def get_costs_per_instance(self, conf1, conf2, aggregate=None, keep_missing=False):
        """
        Get costs for two configurations

        Parameters
        ----------
        conf1, conf2: Configuration
            configs to be compared
        aggregate: Function or None
            used to aggregate cost over different seeds, takes a list as
            argument
        keep_missing: Boolean
            keep instances with one or no configs for it

        Returns
        -------
        instance_costs: dict(instance->tuple(cost1, cost2))
            costs for both configurations
        """
        conf1_cost = self.get_cost_per_instance(conf1, aggregate=aggregate)
        conf2_cost = self.get_cost_per_instance(conf2, aggregate=aggregate)
        instance_cost = dict()
        for i in set(conf1_cost.keys()) | set(conf2_cost.keys()):
            instance_cost[i] = (conf1_cost.get(i), conf2_cost.get(i))

        if not keep_missing:
            before = len(instance_cost)
            instance_cost = {i: instance_cost[i] for i in instance_cost if
                             instance_cost[i][0] and instance_cost[i][1]}
            self.logger.info("Remove %d/%d instances because they are not "
                             "evaluated on both configurations.",
                             before - len(instance_cost), before)
        return instance_cost

    def calculate_par10(self, def_costs, inc_costs):
        """ Calculate par10-values of default and incumbent configs. """
        default = {i:c if c < self.scenario.cutoff else self.scenario.cutoff*10
                   for i, c in def_costs.items()}
        incumbent = {i:c if c < self.scenario.cutoff else self.scenario.cutoff*10
                     for i, c in inc_costs.items()}
        self.def_par10_combined = np.mean(list(default.values()))
        self.inc_par10_combined = np.mean(list(incumbent.values()))
        self.def_par10_train = np.mean([c for i, c in default.items() if i in
                                        self.train_inst])
        self.def_par10_test = np.mean([c for i, c in default.items() if i in
                                       self.test_inst])
        self.inc_par10_train = np.mean([c for i, c in incumbent.items() if i in
                                        self.train_inst])
        self.inc_par10_test = np.mean([c for i, c in incumbent.items() if i in
                                       self.test_inst])

    def create_overview_table(self):
        """ Create overview-table. """
        # TODO: left-align, make first and third column bold
        d = OrderedDict([('Scenario name', self.scenario.output_dir),
                         ('# Train instances', len(self.scenario.train_insts)),
                         ('# Test instances', len(self.scenario.test_insts)),
                         ('# Parameters', len(self.scenario.cs.get_hyperparameters())),
                         ('Cutoff time', self.scenario.cutoff),
                         ('Deterministic', self.scenario.deterministic),
                         ('# Configs evaluated', len(self.runhistory.config_ids))])
        a = []
        keys = list(d.keys())
        half_size_d = len(keys)//2
        for i in range(half_size_d):
            j = i + half_size_d
            a.append((keys[i], d[keys[i]], keys[j], d[keys[j]]))
        if len(keys)%2 == 1:
            a.append((keys[half_size_d], d[keys[half_size_d]], '', ''))
        df = DataFrame(data=a)
        table = df.to_html(header=False, index=False, justify='left')
        return table

    def create_html_table(self):
        """ Create PAR10-table, compare default against incumbent on train-,
        test- and combined instances. """
        array = np.array([[self.def_par10_train, self.def_par10_test, self.def_par10_combined],
                          [self.inc_par10_train, self.inc_par10_test, self.inc_par10_combined]])
        df = DataFrame(data=array, index=['Default', 'Incumbent'],
                       columns=['Train', 'Test', 'Combined'])
        table = df.to_html()
        return table

    def parameter_importance(self):
        """ Calculate parameter-importance using the PIMP-package. """
