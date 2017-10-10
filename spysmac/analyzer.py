import os
import logging
from collections import OrderedDict
import typing
import json

import numpy as np
from pandas import DataFrame
import matplotlib
matplotlib.use('Agg')

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
from spysmac.utils.helpers import get_cost_dict_for_config, get_timeout

__author__ = "Joshua Marben"
__copyright__ = "Copyright 2017, ML4AAD"
__license__ = "3-clause BSD"
__maintainer__ = "Joshua Marben"
__email__ = "joshua.marben@neptun.uni-freiburg.de"

class Analyzer(object):
    """
    Analyze SMAC-output data.
    Compares two configurations (default vs incumbent) over multiple SMAC-runs
    and outputs PAR10, timeouts, scatterplots, parameter importance etc.
    """

    def __init__(self, global_rh, train_test, scenario):
        """Saves all relevant information that arises during the analysis. """
        self.logger = logging.getLogger("spysmac.analyzer")

        self.global_rh = global_rh
        self.train_test = train_test
        self.scenario = scenario

        self.performance_table = None

    def get_timeouts(self, config):
        """ Get number of timeouts in config per runs in total (not per
        instance)

        Parameters
        ----------
        config: Configuration
            configuration from which to calculate the timeouts"""
        cutoff = self.scenario.cutoff
        timeouts = get_timeout(self.global_rh, config, cutoff)
        if self.train_test:
            if not cutoff:
                return (("N","A"),("N","A"))
            train_timeout = len([i for i in timeouts if (timeouts[i] == False
                                  and i in self.scenario.train_insts)])
            train_no_timeout = len([i for i in timeouts if (timeouts[i] == True
                                  and i in self.scenario.train_insts)])
            test_timeout = len([i for i in timeouts if (timeouts[i] == False
                                  and i in self.scenario.test_insts)])
            test_no_timeout = len([i for i in timeouts if (timeouts[i] == True
                                  and i in self.scenario.test_insts)])
            return ((train_timeout, train_no_timeout), (test_timeout, test_no_timeout))
        else:
            if not cutoff:
                return ("N","A")
            timeout = len([i for i in timeouts if timeouts[i] == False])
            no_timeout = len([i for i in timeouts if timeouts[i] == True])
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
            runs = [(k, runs[k]) if runs[k] < self.scenario.cutoff
                        else (k, self.scenario.cutoff*par)
                        for k in runs]
        else:
            runs = [(k, runs[k]) for k in runs]
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

    def fanova(self, incumbent, output, num_params=10):
        """Wrapper for parameter_importance to save the importance-object/
        extract the results. We want to show the top X (10) most important
        parameter-fanova-plots.

        Parameters
        ----------
        incumbent: Configuration
            incumbent configuration
        output: str
            output-dir
        num_params: int
            how many of the top important parameters should be shown

        Returns
        -------
        parameter_imp: list[str]
            list with the most important parameters
        """
        importance = self.parameter_importance("fanova", incumbent, output)
        parameter_imp = importance.evaluator.evaluated_parameter_importance
        parameter_imp = sorted([(k, parameter_imp[k]) for k in
            parameter_imp.keys()], key=lambda x:x[1], reverse=True)
        return parameter_imp[:num_params]

    def parameter_importance(self, modus, incumbent, output):
        """Calculate parameter-importance using the PIMP-package.
        Currently ablation, forward-selection and fanova are used.

        Parameters
        ----------
        modus: str
            modus for parameter importance, from [forward-selection, ablation,
            fanova]

        Returns
        -------
        importance: pimp.Importance
            importance object with evaluated data
        """
        # Get parameter-values and costs per config in arrays.
        configs = self.global_rh.get_all_configs()
        X = np.array([c.get_array() for c in configs])
        y = np.array([self.global_rh.get_cost(c) for c in configs])

        # Evaluate parameter importance
        save_folder = output
        importance = Importance(scenario=self.scenario,
                                runhistory=self.global_rh,
                                incumbent=incumbent,
                                parameters_to_evaluate=len(self.scenario.cs.get_hyperparameters()),
                                save_folder=save_folder,
                                seed=12345)
        result = importance.evaluate_scenario(modus)
        with open(os.path.join(save_folder, 'pimp_values_%s.json' % modus), 'w') as out_file:
            json.dump(result, out_file, sort_keys=True, indent=4, separators=(',', ': '))
        importance.plot_results(name=os.path.join(save_folder, modus), show=False)
        return importance

    def create_overview_table(self, best_folder):
        """ Create overview-table.

        Parameters
        ----------
        best_folder: str
            path to folder/run with best incumbent

        Returns
        -------
        table: str
            overview table in HTML
        """
        overview = OrderedDict([('Run with best incumbent', best_folder),
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

    def create_performance_table(self, default, incumbent):
        """Create table, compare default against incumbent on train-,
        test- and combined instances. Listing PAR10, PAR1 and timeouts.
        Distinguishes between train and test, if available."""
        def_timeout, inc_timeout = self.get_timeouts(default), self.get_timeouts(incumbent)
        def_par10, inc_par10 = self.get_parX(default, 10), self.get_parX(incumbent, 10)
        def_par1, inc_par1 = self.get_parX(default, 1), self.get_parX(incumbent, 1)
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
        self.performance_table = table
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
        default = [default[k] if default[k] != None else "inactive" for k in keys]
        incumbent = [incumbent[k] if incumbent[k] != None else "inactive" for k in keys]
        table = list(zip(keys, default, incumbent))
        # Show first parameters that changed
        same = [x for x in table if x[1] == x[2]]
        diff = [x for x in table if x[1] != x[2]]
        table = []
        if len(diff) > 0:
            table.extend([("-------------- Changed parameters: "\
                           "--------------", "-----", "-----")])
            table.extend(diff)
        if len(same) > 0:
            table.extend([("-------------- Unchanged parameters: "\
                           "--------------", "-----", "-----")])
            table.extend(same)
        keys, table = [k[0] for k in table], [k[1:] for k in table]
        df = DataFrame(data=table, columns=["Default", "Incumbent"], index=keys)
        table = df.to_html()
        return table

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

