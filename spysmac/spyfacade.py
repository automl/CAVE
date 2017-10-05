import os
import logging
from collections import OrderedDict
from contextlib import contextmanager
import typing
import json

import numpy as np
from pandas import DataFrame

from smac.configspace import Configuration
from smac.epm.rf_with_instances import RandomForestWithInstances
from smac.optimizer.objective import average_cost
from smac.runhistory.runhistory import RunKey, RunValue, RunHistory
from smac.runhistory.runhistory2epm import RunHistory2EPM4Cost
from smac.scenario.scenario import Scenario
from smac.utils.io.traj_logging import TrajLogger
from smac.utils.validate import Validator

from pimp.importance.importance import Importance

from spysmac.html.html_builder import HTMLBuilder
from spysmac.plot.plotter import Plotter
from spysmac.smacrun import SMACrun
from spysmac.analyzer import Analyzer
from spysmac.utils.helpers import get_loss_per_instance, get_cost_dict_for_config

__author__ = "Joshua Marben"
__copyright__ = "Copyright 2017, ML4AAD"
__license__ = "3-clause BSD"
__maintainer__ = "Joshua Marben"
__email__ = "joshua.marben@neptun.uni-freiburg.de"

@contextmanager
def changedir(newdir):
    """ Helper function to change directory, for example to create a scenario
    from file, where paths to the instance- and feature-files are relative to
    the original SMAC-execution-directory. Same with target algorithms that need
    be executed for validation. """
    olddir = os.getcwd()
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(olddir)

class SpySMAC(object):
    """
    """

    def __init__(self, folders: typing.List[str], output: str,
                 ta_exec_dir: str='.', missing_data_method: str='validation'):
        """
        Initialize SpySMAC facade to handle analyzing, plotting and building the
        report-page easily.

        Arguments
        ---------
        folders: list<strings>
            paths to relevant SMAC runs
        output: string
            output for spysmac to write results (figures + report)
        ta_exec_dir: string
            execution directory for target algorithm
        missing_data_method: string
            from [validation, epm], how to estimate missing runs
        """
        self.logger = logging.getLogger("spysmac.spyfacade")

        self.ta_exec_dir = ta_exec_dir

        # Create output if necessary
        self.output = output
        self.logger.info("Writing to %s", self.output)
        if not os.path.exists(output):
            self.logger.debug("Output-dir %s does not exist, creating", self.output)
            os.makedirs(output)

        # Global runhistory combines all runs of individual SMAC-runs
        self.global_rh = RunHistory(average_cost)

        # Save all relevant SMAC-runs in a list
        self.runs = []
        for folder in folders:
            if not os.path.exists(folder):
                raise ValueError("The specified SMAC-output in %s doesn't exist.",
                                 folder)
            self.logger.debug("Collecting data from %s.", folder)
            self.runs.append(SMACrun(folder, ta_exec_dir))

        self.scenario = self.runs[0].scen

        # Update global runhistory with all available runhistories
        self.logger.debug("Update global rh with all available rhs!")
        runhistory_fns = [os.path.join(f, "runhistory.json") for f in folders]
        for rh_file in runhistory_fns:
            self.global_rh.update_from_json(rh_file, self.scenario.cs)
        self.logger.debug('Combined number of Runhistory data points: %d. '
                          '# Configurations: %d. # Runhistories: %d',
                          len(runhistory_fns), len(self.global_rh.data),
                          len(self.global_rh.get_all_configs()))

        # Estimate all missing costs using validation or EPM
        self.complete_data(method=missing_data_method)
        self.best_run = min(self.runs, key=lambda run: run.get_incumbent()[1])

        # Check scenarios for consistency in relevant attributes
        # TODO check for consistency in scenarios
        for run in self.runs:
            if not run.scen == self.scenario:
                #raise ValueError("Scenarios don't match up ({})".format(run.folder))
                pass
        self.default = self.scenario.cs.get_default_configuration()
        self.incumbent = self.best_run.get_incumbent()[0]

        # Following variable determines whether a distinction is made
        self.train_test = bool(self.scenario.train_insts != [None] and
                               self.scenario.test_insts != [None])

        self.analyzer = Analyzer(self.global_rh, self.train_test, self.scenario)
        conf1_runs = get_cost_dict_for_config(self.global_rh, self.default)
        conf2_runs = get_cost_dict_for_config(self.global_rh, self.incumbent)
        self.plotter = Plotter(self.scenario, self.train_test, conf1_runs, conf2_runs)
        self.website = OrderedDict([])

    def complete_data(self, method="validation"):
        """Complete missing data of runs to be analyzed. Either using validation
        or EPM.
        """
        with changedir(self.ta_exec_dir):
            self.logger.info("Completing data using %s.", method)

            for run in self.runs:
                validator = Validator(run.scen, run.traj, "")

                if method == "validation":
                    # TODO determine # repetitions
                    run.runhistory = validator.validate('def+inc', 'train+test', 1, -1,
                                                runhistory=self.global_rh)
                elif method == "epm":
                    run.runhistory = validator.validate_epm('def+inc', 'train+test', 1,
                                                    runhistory=self.global_rh)
                else:
                    raise ValueError("Missing data method illegal (%s)",
                                     method)

                self.global_rh.update(run.runhistory)

    def analyze(self,
                performance=False, cdf=False, scatter=False, confviz=False,
                forward_selection=False, ablation=False, fanova=False):
        """Analyze the available data and build HTML-webpage as dict.
        Save webpage in 'self.output/SpySMAC/report.html'.
        Analyzing is performed with the analyzer-instance that is initialized in
        the __init__, same with plotting and plotter-instance..

        Parameters
        ----------
        par10: bool
            whether to calculate par10-values
        cdf: bool
            whether to plot cdf
        scatter: bool
            whether to plot scatter
        forward_selection: bool
            whether to apply forward selection
        ablation: bool
            whether to apply ablation
        fanova: bool
            whether to apply fanova
        """
        overview = self.analyzer.create_overview_table(self.best_run.folder)
        self.website["Meta Data"] = {
                     "table": overview,
                     "tooltip": "Meta data such as number of instances, "
                                "parameters, general configurations..." }

        best_config = self.analyzer.config_to_html(self.default, self.incumbent)
        self.website["Best configuration"] = {"table": best_config,
                     "tooltip": "Comparing parameters of default and incumbent."}

        if performance:
            performance_table = self.analyzer.create_performance_table(
                                self.default, self.incumbent)
            self.website["Performance"] = {"table": performance_table}

        if cdf:
            cdf_path = os.path.join(self.output, 'cdf.png')
            self.plotter.plot_cdf_compare(output=cdf_path)
            self.website["Cumulative distribution function (CDF)"] = {
                     "figure": cdf_path,
                     "tooltip": "Plotting default vs incumbent on the time to "
                                "solve instances. Timeouts are excluded."}

        if scatter and (self.scenario.train_insts != [[None]]):
            scatter_path = os.path.join(self.output, 'scatter.png')
            self.plotter.plot_scatter(output=scatter_path)
            self.website["Scatterplot"] = {
                     "figure" : scatter_path,
                     "tooltip": "Plot all instances in direct comparison of "
                                "default and incumbent costs."}
        elif scatter:
            self.logger.info("Scatter plot desired, but no instances available.")

        if  confviz and self.scenario.feature_array is not None:
            confviz = self.plotter.visualize_configs(self.scenario, self.global_rh)
            self.website["Configuration Visualization"] = {
                    "table" : confviz,
                    "tooltip" : "Using PCA to reduce dimensionality of the "
                                "search space  and plot the distribution of"
                                "evaluated configurations. The bigger the dot, "
                                "the more often the configuration was "
                                "evaluated. The colours refer to the predicted "
                                "performance in that part of the search space."}
        elif confviz:
            self.logger.info("Configuration visualization desired, but no "
                             "instance-features available.")

        # PARAMETER IMPORTANCE
        if ablation or forward_selection or fanova:
            self.website["Parameter Importance"] = OrderedDict([])
        if ablation:
            self.analyzer.parameter_importance("ablation", self.incumbent,
                                               self.output)
            ablationpercentage_path = os.path.join(self.output, "ablationpercentage.png")
            ablationperformance_path = os.path.join(self.output, "ablationperformance.png")
            self.website["Parameter Importance"]["Ablation (percentage)"] = {
                        "figure": ablationpercentage_path}
            self.website["Parameter Importance"]["Ablation (performance)"] = {
                        "figure": ablationperformance_path}
        if forward_selection:
            self.analyzer.parameter_importance("forward-selection", self.incumbent,
                                               self.output)
            f_s_barplot_path = os.path.join(self.output, "forward-selection-barplot.png")
            f_s_chng_path = os.path.join(self.output, "forward-selection-chng.png")
            self.website["Parameter Importance"]["Forward Selection (barplot)"] = {
                        "figure": f_s_barplot_path}
            self.website["Parameter Importance"]["Forward Selection (chng)"] = {
                        "figure": f_s_chng_path}
        if fanova:
            self.analyzer.parameter_importance("fanova", self.incumbent, self.output)

        builder = HTMLBuilder(self.output, "SpySMAC")
        builder.generate_html(self.website)


    def _eq_scenarios(self, scen1: Scenario, scen2: Scenario):
        """Custom function to compare relevant features of scenarios.

        Parameters
        ----------
        scen1, scen2 -- Scenario
            scenarios to be compared
        """
        relevant = ["train_insts", "test_insts", "cs", "features_dict",
                    "initial_incumbent", "cutoff", "cost_for_crash"]
        #for member in 

