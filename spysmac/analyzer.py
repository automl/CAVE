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

class Analyzer(object):
    """
    Analyze SMAC-output data.
    Compares two configurations (default vs incumbent) over multiple SMAC-runs
    and outputs PAR10, timeouts, scatterplots, parameter importance etc.
    """

    def __init__(self, folders: typing.List[str], output: str,
                 ta_exec_dir: str='.', missing_data_method: str='validation'):
        """
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
        self.logger = logging.getLogger("spysmac.analyzer")

        self.missing_data_method = missing_data_method
        self.ta_exec_dir = ta_exec_dir

        self.folders = folders
        self.logger.debug("Folders: %s", str(self.folders))

        # Create output if necessary
        self.output = output
        self.logger.info("Writing to %s", self.output)
        if not os.path.exists(output):
            self.logger.info("Output-dir %s does not exist, creating", self.output)
            os.makedirs(output)

        # Global runhistory combines all runs of individual SMAC-runs
        self.global_rh = RunHistory(average_cost)

        # Save all relevant SMAC-runs in a list
        self.runs = []
        for folder in self.folders:
            if not os.path.exists(folder):
                raise ValueError("The specified SMAC-output in %s doesn't exist.",
                                 folder)
            self.logger.debug("Collecting data from %s.", folder)
            self.runs.append(SMACrun(folder, ta_exec_dir))

        self.scenario = self.runs[0].scen

        # Update global runhistory with all available runhistories
        self.logger.debug("Update global rh with all available rhs!")
        runhistory_fns = [os.path.join(f, "runhistory.json") for f in self.folders]
        for rh_file in runhistory_fns:
            self.global_rh.update_from_json(rh_file, self.scenario.cs)
        self.logger.debug('Combined number of Runhistory data points: %d. '
                          '# Configurations: %d. # Runhistories: %d',
                          len(runhistory_fns), len(self.global_rh.data),
                          len(self.global_rh.get_all_configs()))

        # Estimate all missing costs using validation or EPM
        self.complete_data()
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

        # Dict to save par10 in as tuple (train, test)
        self.par10 = {}

        # Paths
        self.paths = {'scatter_path' : False,
                      'cdf_combined_path' : False,
                      'f_s_barplot_path' : os.path.join(self.output, "forward-selection-barplot.png"),
                      'f_s_chng_path' : os.path.join(self.output, "forward-selection-chng.png"),
                      'ablationpercentage_path' : os.path.join(self.output, "ablationpercentage.png"),
                      'ablationperformance_path' : os.path.join(self.output, "ablationperformance.png")}

    def complete_data(self):
        """Complete missing data of runs to be analyzed. Either using validation
        or EPM.
        """
        with changedir(self.ta_exec_dir):
            self.logger.info("Completing data using %s.", self.missing_data_method)

            for run in self.runs:
                validator = Validator(run.scen, run.traj, "")

                if self.missing_data_method == "validation":
                    # TODO determine # repetitions
                    run.rh = validator.validate('def+inc', 'train+test', 1, -1,
                                                runhistory=self.global_rh)
                elif self.missing_data_method == "epm":
                    run.rh = validator.validate_epm('def+inc', 'train+test', 1,
                                                    runhistory=self.global_rh)
                else:
                    raise ValueError("Missing data method illegal (%s)",
                                     self.missing_data_method)

                self.global_rh.update(run.rh)

    def analyze(self, par10=True, cdf=True, scatter=True, confviz=True,
                forward_selection=True, ablation=True, fanova=True):
        """
        Performs analysis of scenario by scrutinizing the runhistory.

        Creates:
            - PAR10-values for default and incumbent (best of all runs)
            - CDF-plot for default and incumbent (best of all runs)
            - Scatter-plot for default and incumbent (best of all runs)
            - Importance (forward-selection, ablation, fanova)
            - (TODO) Search space heat map
            - (TODO) Parameter search space flow map

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
        # TODO loss to cost
        default_loss_per_inst = get_loss_per_instance(self.global_rh,
                                                      self.default, aggregate=np.mean)
        incumbent_loss_per_inst = get_loss_per_instance(self.global_rh,
                                                        self.incumbent, aggregate=np.mean)
        if not len(default_loss_per_inst) == len(incumbent_loss_per_inst):
            self.logger.warning("Default evaluated on %d instances, "
                                "incumbent evaluated on %d instances! "
                                "Might lead to unexpected results, consider "
                                "validating your results.",
                                len(default_loss_per_inst), len(incumbent_loss_per_inst))

        # Create table with basic information on scenario and runs
        self.overview = self.create_overview_table()

        # Analysis

        # Plotting
        conf1_runs = get_cost_dict_for_config(self.global_rh, self.default)
        conf2_runs = get_cost_dict_for_config(self.global_rh, self.incumbent)
        plotter = Plotter(self.scenario, self.train_test, conf1_runs,
                conf2_runs)
        if scatter and (self.scenario.train_insts != [[None]]):
            scatter_path = os.path.join(self.output, 'scatter.png')
            plotter.plot_scatter(output=scatter_path)
            self.paths['scatter_path'] = scatter_path
        elif scatter:
            self.logger.info("Scatter plot desired, but no instances available.")

        if cdf:
            cdf_path = os.path.join(self.output, 'cdf.png')
            plotter.plot_cdf_compare(output=cdf_path)
            self.paths['cdf_path'] = cdf_path

        # Visualizing configurations (via plotter)
        self.confviz = None
        if self.scenario.feature_array is not None and confviz: #confviz:
            self.confviz = plotter.visualize_configs(self.scenario, self.global_rh)
        elif confviz:
            self.logger.info("Configuration visualization desired, but no "
                             "instance-features available.")

        # PARAMETER IMPORTANCE
        if ablation:
            self.parameter_importance("ablation")
        if forward_selection:
            self.parameter_importance("forward-selection")
        if fanova:
            self.parameter_importance("fanova")

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
                    {"table": self.overview,
                     "tooltip": "Meta data such as number of instances, "
                                "parameters, general configurations..." }),
                   ("Best configuration",
                    {"table":
                        self.config_to_html(self.default, self.incumbent),
                     "tooltip": "Comparing parameters of default and incumbent."})])

        #if self.default in self.par10 and self.incumbent in self.par10:
        par10_table = self.create_performance_table()
        website["Performance"] = {"table": par10_table}

        if self.paths['scatter_path']:
            website["Scatterplot"] = {
                     "figure" : self.paths['scatter_path'],
                     "tooltip": "Plot all instances in direct comparison of "
                                "default and incumbent costs."}

        if self.paths['cdf_path']:
            website["Cumulative distribution function (CDF)"] = {
                     "figure": self.paths['cdf_path'],
                     "tooltip": "Plotting default vs incumbent on the time to "
                                "solve instances. Timeouts are excluded."}
        website["Parameter Importance"] = OrderedDict([
                       ("Forward Selection (barplot)",
                        {"figure": self.paths['f_s_barplot_path']}),
                       ("Forward Selection (chng)",
                        {"figure": self.paths['f_s_chng_path']}),
                       ("Ablation (percentage)",
                        {"figure": self.paths['ablationpercentage_path']}),
                       ("Ablation (performance)",
                        {"figure": self.paths['ablationperformance_path']})])

        if self.confviz:
            website["Configuration Visualization"] = {"table" :
                               self.confviz}
        builder.generate_html(website)
        return website

    def get_timeouts(self, config):
        """ Get number of timeouts in config per runs in total (not per
        instance) """
        costs = get_cost_dict_for_config(self.global_rh, config, metric='time')
        cutoff = self.scenario.cutoff
        if self.train_test:
            if not cutoff:
                return (("N/A","N/A"),("N/A","N/A"))
            train_timeout, test_timeout = 0, 0
            train_no_timeout, test_no_timeout = 0, 0
            for run in costs:
                if (cutoff and run.instance in self.scenario.train_insts and
                        costs[run] >= cutoff):
                    train_timeout += 1
                elif (cutoff and run.instance in self.scenario.train_insts and
                        costs[run] < cutoff):
                    train_no_timeout += 1
                if (cutoff and run.instance in self.scenario.test_insts and
                        costs[run] >= cutoff):
                    test_timeout += 1
                elif (cutoff and run.instance in self.scenario.test_insts and
                        costs[run] < cutoff):
                    test_no_timeout += 1
            return ((train_timeout, train_no_timeout), (test_timeout, test_no_timeout))
        else:
            if not cutoff:
                return ("N/A","N/A")
            timeout, no_timeout = 0, 0
            for run in costs:
                if cutoff and costs[run] >= cutoff:
                    timeout += 1
                else:
                    no_timeout += 1
            return (timeout, no_timeout)

    def get_parX(self, config, par=10):
        """Calculate parX-values of default and incumbent configs.
        First determine PAR-timeouts for each run on each instances,
        Second average over train/test if available, else just average.

        Parameters
        ----------
        config: Configuration
            config to be calculated
        par: int
            par-factor to use

        Returns
        -------
        (train, test) OR average -- tuple<float, float> OR float
            PAR10 values for train- and test-instances, if available as tuple
            else the general average
        """
        runs = get_cost_dict_for_config(self.global_rh, config)
        # Penalize
        if self.scenario.cutoff:
            runs = [(k.instance, runs[k]) if runs[k] < self.scenario.cutoff
                        else (k.instance, self.scenario.cutoff*par)
                        for k in runs]
        else:
            runs = [(k.instance, runs[k]) for k in runs]
            self.logger.info("Calculating penalized average runtime without "
                             "cutoff...")

        # Average
        if self.train_test:
            train = np.mean([c for i, c in runs if i in
                             self.scenario.train_insts])
            test = np.mean([c for i, c in runs if i in
                            self.scenario.test_insts])
            return (train, test)
        else:
            return np.mean([c for i, c in runs])

    def parameter_importance(self, modus):
        """Calculate parameter-importance using the PIMP-package.
        Currently ablation, forward-selection and fanova are used.

        Parameters
        ----------
        modus: str
            modus for parameter importance, from [forward-selection, ablation,
            fanova]
        """
        # Get parameter-values and costs per config in arrays.
        configs = self.global_rh.get_all_configs()
        X = np.array([c.get_array() for c in configs])
        y = np.array([self.global_rh.get_cost(c) for c in configs])

        # Evaluate parameter importance
        save_folder = self.output
        importance = Importance(scenario=self.scenario,
                                runhistory=self.global_rh,
                                incumbent=self.incumbent,
                                parameters_to_evaluate=len(self.scenario.cs.get_hyperparameters()),
                                save_folder=save_folder,
                                seed=12345)
        result = importance.evaluate_scenario(modus)
        with open(os.path.join(save_folder, 'pimp_values_%s.json' % modus), 'w') as out_file:
            json.dump(result, out_file, sort_keys=True, indent=4, separators=(',', ': '))
        importance.plot_results(name=os.path.join(save_folder, modus), show=False)

    def create_overview_table(self):
        """ Create overview-table. """
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
        #TODO timeouts over runs or over instances? instance-average over PAR1
        # or PAR10?
        def_timeout, inc_timeout = self.get_timeouts(self.default), self.get_timeouts(self.incumbent)
        def_par10, inc_par10 = self.get_parX(self.default, 10), self.get_parX(self.incumbent, 10)
        def_par1, inc_par1 = self.get_parX(self.default, 1), self.get_parX(self.incumbent, 1)
        dec_place = 3
        if self.train_test:
            # Distinction between train and test
            # Create table
            array = np.array([[round(def_par10[0], dec_place),
                               round(def_par10[1], dec_place),
                               round(inc_par10[0], dec_place),
                               round(inc_par10[1], dec_place)],
                              [round(def_par1[0], dec_place),
                               round(def_par1[1], dec_place),
                               round(inc_par1[0], dec_place),
                               round(inc_par1[1], dec_place)],
                              ["{}/{}".format(def_timeout[0][0], def_timeout[0][1]),
                               "{}/{}".format(def_timeout[1][0], def_timeout[1][1]),
                               "{}/{}".format(inc_timeout[0][0], inc_timeout[0][1]),
                               "{}/{}".format(inc_timeout[1][0], inc_timeout[1][1])
                               ]])
            df = DataFrame(data=array, index=['PAR10', 'PAR1', 'Timeouts'],
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
            table = new_table + table
        else:
            # No distinction between train and test
            array = np.array([[round(def_par10, dec_place),
                               round(inc_par10, dec_place)],
                              [round(def_par1, dec_place),
                               round(inc_par1, dec_place)],
                              ["{}/{}".format(def_timeout[0], def_timeout[1]),
                               "{}/{}".format(inc_timeout[0], inc_timeout[1])]])
            df = DataFrame(data=array, index=['PAR10', 'PAR1', 'Timeouts'],
                           columns=['Default', 'Incumbent'])
            table = df.to_html()
        return table

    def config_to_html(self, default: Configuration, incumbent: Configuration):
        """Create HTML-table to compare Configurations.
        Removes unused parameters.

        Parameters
        ----------
        default, incumbent: Configurations
            configurations to be converted

        Returns
        -------
        table: str
            HTML-table comparing default and incumbent
        """
        # Remove unused parameters
        keys = [k for k in default.keys() if default[k] or incumbent[k]]
        default = [default[k] for k in keys]
        incumbent = [incumbent[k] for k in keys]
        table = list(zip(default, incumbent))
        df = DataFrame(data=table, columns=["Default", "Incumbent"], index=keys)
        table = df.to_html()
        return table

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

    def _split_table(self, table: OrderedDict):
        """Splits an OrderedDict into a list of tuples that can be turned into a
        HTML-table with pandas DataFrame

        Parameters
        ----------
        table: OrderedDict
            table that is to be split into two columns

        Returns
        -------
        table_split: List[tuple(key, value, key, value)]
            list with two key-value pairs per entry that can be used by pandas
            df.to_html()
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

