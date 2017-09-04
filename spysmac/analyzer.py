import os
import logging as log
from collections import OrderedDict
from contextlib import contextmanager

import numpy as np
from pandas import DataFrame

from smac.optimizer.objective import average_cost
from smac.runhistory.runhistory import RunKey, RunValue, RunHistory
from smac.utils.io.traj_logging import TrajLogger
from smac.utils.validate import Validator

#from pimp.importance.importance import Importance
#from pimp.pimp import PIMP

from spysmac.html.html_builder import HTMLBuilder
from spysmac.plot.plotter import Plotter
from spysmac.smacrun import SMACrun


__author__ = "Joshua Marben"
__copyright__ = "Copyright 2017, ML4AAD"
__license__ = "3-clause BSD"
__maintainer__ = "Joshua Marben"
__email__ = "joshua.marben@neptun.uni-freiburg.de"

@contextmanager
def changedir(newdir):
    olddir = os.getcwd()
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(olddir)

class Analyzer(object):
    """
    Analyze SMAC-output data.
    Compares two configurations (default vs incumbent) over multiple SMAC-runs
    and outputs PAR10, timeouts, scatterplots, etc.
    """

    def __init__(self, folders, output, ta_exec_dir='.'):
        """
        Arguments
        ---------
        folder: list<strings>
            paths to relevant SMAC runs
        output: string
            output for spysmac to write results (figures + report)
        """
        self.logger = log.getLogger("spysmac.analyzer")

        # Create output if necessary
        self.output = output
        self.logger.info("Writing to %s", self.output)
        if not os.path.exists(output):
            self.logger.info("Output-dir %s does not exist, creating", self.output)
            os.makedirs(output)

        # Global runhistory combines all runs of individual SMAC-runs to avoid
        # recalculation of configurations (such as default)
        self.global_rh = RunHistory(average_cost)
        self.ta_exec_dir = ta_exec_dir

        # Save all relevant SMAC-runs in a list and validate them
        # TODO check for consistency in scenarios
        self.runs = []
        with changedir(ta_exec_dir):
            for folder in folders:
                self.logger.debug("Collecting data from %s.", folder)
                self.runs.append(SMACrun(folder, self.global_rh))
            for run in self.runs:
                self.global_rh.update(run.validate())
                self.global_rh.update_from_json(run.rh_fn, run.scen.cs)
            self.best_run = min(self.runs, key=lambda run: run.get_incumbent()[1])
            self.scenario = self.best_run.scen
            # Check scenarios for consistency in relevant attributes
            for run in self.runs:
                if not run.scen == self.scenario:
                    #raise ValueError("Scenarios don't match up ({})".format(run.folder))
                    pass
        self.default = self.scenario.cs.get_default_configuration()

        # Paths
        self.scatter_path = os.path.join(self.output, 'scatter.png')
        self.cdf_combined_path = os.path.join(self.output, 'def_inc_cdf_comb.png')

    def analyze(self):
        """
        Performs analysis of scenario by scrutinizing the runhistory.

        Creates:
            - PAR10-values for default and incumbent (best of all runs)
            - CDF-plot for default and incumbent (best of all runs)
            - Scatter-plot for default and incumbent (best of all runs)
            - (TODO) Importance
            - (TODO) Search space heat map
            - (TODO) Parameter search space flow map
        """
        default_loss_per_inst = self.best_run.get_loss_per_instance(self.default, aggregate=np.mean)
        inc = self.best_run.get_incumbent()[0]
        incumbent_loss_per_inst = self.best_run.get_loss_per_instance(inc, aggregate=np.mean)
        if not len(default_loss_per_inst) == len(incumbent_loss_per_inst):
            self.logger.warning("Default evaluated on %d instances, "
                                "incumbent evaluated on %d instances! "
                                "Might lead to unexpected results, consider "
                                "re-validating your results.",
                                len(default_loss_per_inst), len(incumbent_loss_per_inst))

        # Create table with basic information on scenario and runs
        self.overview = self.create_overview_table()

        # Analysis
        self.logger.debug("Calculate par10-values")
        def_par10 = self.calculate_par10(default_loss_per_inst)
        self.def_par10_train, self.def_par10_test, self.def_par10_combined = def_par10
        inc_par10 = self.calculate_par10(incumbent_loss_per_inst)
        self.inc_par10_train, self.inc_par10_test, self.inc_par10_combined = inc_par10
        self.par10_table = self.create_performance_table()

        # Plotting
        plotter = Plotter()
        # Scatterplot
        self.logger.debug("Plot scatter")
        plotter.plot_scatter(default_loss_per_inst, incumbent_loss_per_inst,
                             output=self.scatter_path,
                             timeout=self.scenario.cutoff)
        # CDF, once with a shared axis, once two separate
        self.logger.debug("Plot CDF")
        loss_dict = {'default' : default_loss_per_inst, 'incumbent' : incumbent_loss_per_inst}
        plotter.plot_cdf_compare(loss_dict,
                                 timeout= self.scenario.cutoff,
                                 #train=self.train_inst, test=self.test_inst,
                                 output=self.cdf_combined_path)

        #self.parameter_importance()

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
                   ("Meta Data",
                    {"table": self.overview}),
                   ("Best configuration",
                    {"table":
                        self.config_to_html(self.default, self.best_run.get_incumbent()[0])}),
                   ("PAR10",
                    {"table": self.par10_table}),
                   ("Scatterplot",
                    {"figure" : self.scatter_path}),
                   ("Cumulative distribution function (CDF)",
                    {"figure": self.cdf_combined_path}),
                   #("Parameter Importance" :
                  ])
        builder.generate_html(website)
        return website


    def calculate_par10(self, losses):
        """ Calculate par10-values of default and incumbent configs.

        Parameters
        ----------
        losses -- dict<str->float>
            mapping of instance to loss

        Returns
        -------
        (train, test, combined) -- tuple<float, float, float>
            PAR10 values for train-, test- and combined instances
        """
        losses = {i:c if c < self.scenario.cutoff else self.scenario.cutoff*10
                   for i, c in losses.items()}
        train = np.mean([c for i, c in losses.items() if i in
                         self.scenario.train_insts])
        test = np.mean([c for i, c in losses.items() if i in
                        self.scenario.test_insts])
        combined = np.mean(list(losses.values()))
        return (train, test, combined)

    def create_overview_table(self):
        """ Create overview-table. """
        # TODO: left-align, make first and third column bold
        overview = OrderedDict([('Run with best incumbent', self.best_run.folder),
                         ('# Train instances', len(self.scenario.train_insts)),
                         ('# Test instances', len(self.scenario.test_insts)),
                         ('# Parameters', len(self.scenario.cs.get_hyperparameters())),
                         ('Cutoff', self.scenario.cutoff),
                         ('Walltime budget', self.scenario.wallclock_limit),
                         ('Runcount budget', self.scenario.ta_run_limit),
                         ('CPU budget', self.scenario.algo_runs_timelimit),
                         ('Deterministic', self.scenario.deterministic),
                         ])
        # Split into two columns
        overview_split = self._split_table(overview)
        # Convert to HTML
        df = DataFrame(data=overview_split)
        table = df.to_html(escape=False, header=False, index=False, justify='left')
        return table

    def create_performance_table(self):
        """ Create PAR10-table, compare default against incumbent on train-,
        test- and combined instances. """
        array = np.array([[self.def_par10_train, self.def_par10_test,
                           self.inc_par10_train, self.inc_par10_test]])
        df = DataFrame(data=array, index=['PAR10'],
                       columns=['Train', 'Test', 'Train', 'Test'])
        table = df.to_html()
        # Insert two-column-header
        table = table.split(sep='</thead>', maxsplit=1)[1]
        new_table = "<table border=\"3\" class=\"dataframe\">\n"\
                    "  <col>\n"\
                    "  <colgroup span=\"2\"></colgroup>\n"\
                    "  <colgroup span=\"2\"></colgroup>\n"\
                    "  <thead>\n"\
                    "    <tr>\n"\
                    "      <td rowspan=\"2\"></td>\n"\
                    "      <th colspan=\"2\" scope=\"colgroup\">Default</th>\n"\
                    "      <th colspan=\"2\" scope=\"colgroup\">Incumbent</th>\n"\
                    "    </tr>\n"\
                    "    <tr>\n"\
                    "      <th scope=\"col\">Train</th>\n"\
                    "      <th scope=\"col\">Test</th>\n"\
                    "      <th scope=\"col\">Train</th>\n"\
                    "      <th scope=\"col\">Test</th>\n"\
                    "    </tr>\n"\
                    "</thead>\n"
        return new_table + table

    def config_to_html(self, default, incumbent):
        """Create HTML-table from Configurations. Removes unused parameters.

        Parameters
        ----------
        default, incumbent: Configurations
            configurations to be converted
        """
        # Remove unused parameters
        keys = [k for k in default.keys() if default[k] or incumbent[k]]
        default = [default[k] for k in keys]
        incumbent = [incumbent[k] for k in keys]
        table = list(zip(default, incumbent))
        df = DataFrame(data=table, columns=["Default", "Incumbent"], index=keys)
        table = df.to_html()
        return table

    def parameter_importance(self):
        """ Calculate parameter-importance using the PIMP-package. """
        configs = self.global_rh.get_all_configs()
        X = np.array([c.get_array() for c in configs])
        y = np.array([self.global_rh.get_cost(c) for c in configs])
        log.info(len(configs))
        with changedir(self.ta_exec_dir):
            pimp = PIMP(scenario=self.scenario, X=X, y=y,
                mode='forward-selection')
            pimp.plot_results(pimp.compute_importances())

    def _eq_scenarios(self, scen1, scen2):
        """Custom function to compare relevant features of scenarios.

        Parameters
        ----------
        scen1, scen2 -- Scenario
            scenarios to be compared
        """
        relevant = ["train_insts", "test_insts", "cs", "features_dict",
                    "initial_incumbent", "cutoff", "cost_for_crash"]
        #for member in 

    def _split_table(self, table):
        """Splits an OrderedDict into a list of tuples that can be turned into a
        HTML-table with pandas DataFrame

        Parameters
        ----------
        table: OrderedDict
            table that is to be split into two columns
        """
        table_split = []
        keys = list(table.keys())
        half_size = len(keys)//2
        for i in range(half_size):
            j = i + half_size
            table_split.append(("<b>"+keys[i]+"</b>", table[keys[i]],
                                "<b>"+keys[j]+"</b>", table[keys[j]]))
        if len(keys)%2 == 1:
            table_split.append(("<b>"+keys[-1]+"</b>", table[keys[-1]], '', ''))
        return table_split

