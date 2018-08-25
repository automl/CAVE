import os
import logging
from collections import OrderedDict

from pandas import DataFrame
import numpy as np

from cave.analyzer.base_analyzer import BaseAnalyzer
from cave.html.html_helpers import figure_to_html

class OverviewTable(BaseAnalyzer):
    def __init__(self, scenario, orig_rh, best_run, num_runs, default, incumbent, output_dir):
        """ Create overview-table.

        Parameters
        ----------
        orig_rh: RunHistory
            runhistory to take stats from
        best_run: ConfiguratorRun
            configurator run object with best incumbent
        num_runs: int
            number of configurator runs

        Returns
        -------
        table: str
            overview table in HTML
        """
        self.logger = logging.getLogger(self.__module__ + '.' + self.__class__.__name__)

        self.output_dir = output_dir

        all_confs = best_run.original_runhistory.get_all_configs()
        num_configs = len(all_confs)
        num_conf_runs = num_runs
        ta_runtime = np.sum([orig_rh.get_cost(conf) for conf in all_confs])
        ta_evals = [len(orig_rh.get_runs_for_config(conf)) for conf in all_confs]
        ta_evals_d = len(orig_rh.get_runs_for_config(default))
        ta_evals_i = len(orig_rh.get_runs_for_config(incumbent))
        min_ta_evals, max_ta_evals, = np.min(ta_evals), np.max(ta_evals)
        mean_ta_evals, ta_evals = np.mean(ta_evals), np.sum(ta_evals)
        num_changed_params = len([p for p in scenario.cs.get_hyperparameter_names()
                                  if best_run.default[p] != best_run.incumbent[p]])
        # Instances
        num_train = len([i for i in scenario.train_insts if i])
        num_test = len([i for i in scenario.test_insts if i])
        # Features
        num_feats = scenario.n_features if scenario.feature_dict else 0
        num_dup_feats = 0
        if scenario.feature_dict:
            dup_feats = DataFrame(scenario.feature_array)
            num_dup_feats = len(dup_feats[dup_feats.duplicated()])  # only contains train instances

        overview = OrderedDict([('Run with best incumbent', os.path.basename(best_run.folder)),
                                # Constants for scenario
                                ('# Train instances', num_train),
                                ('# Test instances', num_test),
                                ('# Parameters', len(scenario.cs.get_hyperparameters())),
                                ('# Features', num_feats),
                                ('# Duplicate Feature vectors', num_dup_feats),
                                ('empty1', 'empty1'),
                                ('# Evaluated Configurations', num_configs),
                                ('# Default evaluations', ta_evals_d),
                                ('# Incumbent evaluations', ta_evals_i),
                                ('Budget spent evaluating configurations', ta_runtime),
                                ('# Changed parameters', num_changed_params),
                                # BREAK
                                ('Cutoff', scenario.cutoff),
                                ('Walltime budget', scenario.wallclock_limit),
                                ('Runcount budget', scenario.ta_run_limit),
                                ('CPU budget', scenario.algo_runs_timelimit),
                                ('Deterministic', scenario.deterministic),
                                ('empty2', 'empty2'),
                                ('empty3', 'empty3'),
                                ('# Runs per Config (min)', min_ta_evals),
                                ('# Runs per Config (mean)', mean_ta_evals),
                                ('# Runs per Config (max)', max_ta_evals),
                                ('Total number of configuration runs', ta_evals),
                                ('Number of configurator runs', num_conf_runs),
                                ])
        # Split into two columns
        overview_split = self._split_table(overview)
        # Convert to HTML
        df = DataFrame(data=overview_split)
        self.table = df
        self.html_table = df.to_html(escape=False, header=False, index=False, justify='left')
        # Insert empty lines
        for i in range(10):
            self.html_table = self.html_table.replace('empty'+str(i), '&nbsp')

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
        half_size = len(keys) // 2
        for i in range(half_size):
            j = i + half_size
            table_split.append(("<b>" + keys[i] + "</b>", table[keys[i]],
                                "<b>" + keys[j] + "</b>", table[keys[j]]))
        if len(keys) % 2 == 1:
            table_split.append(("<b>"+keys[-1]+"</b>", table[keys[-1]], '', ''))
        return table_split

    def get_table(self):
        return self.table

    def get_html(self, d=None):
        if d is not None:
            d["Overview"] = {"table": self.html_table}
        return self.html_table

    def get_jupyter(self):
        from IPython.core.display import HTML, display
        display(HTML(self.get_html()))

