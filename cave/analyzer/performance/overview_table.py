import os
from collections import OrderedDict

import numpy as np
from ConfigSpace.hyperparameters import NumericalHyperparameter, CategoricalHyperparameter, OrdinalHyperparameter, \
    Constant
from pandas import DataFrame

from cave.analyzer.base_analyzer import BaseAnalyzer
from cave.utils.helpers import get_config_origin


class OverviewTable(BaseAnalyzer):
    """
    Meta data, i.e. number of instances and parameters as well as configuration budget. Statistics apply to the
    best run, if multiple configurator runs are compared.
    """
    def __init__(self, runscontainer):
        super().__init__(runscontainer)
        self.output_dir = runscontainer.output_dir

        self.run()

    def get_name(self):
        return "Meta Data"

    def run(self):
        """ Generate tables. """
        scenario = self.runscontainer.scenario

        # General infos
        general_dict = self._general_dict(scenario)
        html_table_general = DataFrame(data=OrderedDict([('General', general_dict)]))
        html_table_general = html_table_general.reindex(list(general_dict.keys()))
        html_table_general = html_table_general.to_html(escape=False, header=False, justify='left')
        self.result["General"] = {"table": html_table_general,
                                  "tooltip": "General information about the optimization scenario."}

        # Run-specific / budget specific infos
        for mode in ['parallel', 'budget']:
            runspec_dict = self._runspec_dict(identify=mode)
            if not runspec_dict:
                continue
            order_spec = list(list(runspec_dict.values())[0].keys())  # Get keys of any sub-dict for order
            html_table_specific = DataFrame(runspec_dict)
            html_table_specific = html_table_specific.reindex(order_spec)
            html_table_specific = html_table_specific.to_html(escape=False, justify='left')
            if mode == 'parallel':
                self.result["Parallel Runs"] = {"table": html_table_specific,
                                                "tooltip": "Information to individual parallel runs."}
            if mode == 'budget':
                self.result["Budgets"] = {"table": html_table_specific,
                                          "tooltip": "Statistics related to the budgets used in this optimization."}

        # ConfigSpace in tabular form
        cs_dict = self._configspace(scenario.cs)
        cs_table = DataFrame(data=cs_dict)
        html_table_cs = cs_table.to_html(escape=False, justify='left', index=False)
        self.result["Configuration Space"] = {"table": html_table_cs,
                                              "tooltip": "The parameter configuration space. "
                                                         "(See github.com/automl/ConfigSpace)"}

        return self.result

    def _general_dict(self, scenario):
        """ Generate the meta-information that holds for all runs (scenario info etc)

        Parameters
        ----------
        scenario: smac.Scenario
            scenario file to get information from
        """
        # general stores information that holds for all runs, runspec holds information on a run-basis
        general = OrderedDict()

        if len(self.runscontainer.get_budgets()) > 1:
            general['# budgets'] = len(self.runscontainer.get_budgets())
        if len(self.runscontainer.get_folders()) > 1:
            general['# parallel runs'] = len(self.runscontainer.get_folders())

        # Scenario related
        general['# parameters'] = len(scenario.cs.get_hyperparameters())
        general['Deterministic target algorithm'] = scenario.deterministic
        general['Optimized run objective'] = scenario.run_obj
        if scenario.cutoff or scenario.run_obj == 'runtime':
            general['Cutoff'] = scenario.cutoff
        if any([str(lim)!='inf' for lim in [scenario.wallclock_limit, scenario.ta_run_limit, scenario.algo_runs_timelimit]]):
            general['Walltime budget'] = scenario.wallclock_limit
            general['Runcount budget'] = scenario.ta_run_limit
            general['CPU budget'] = scenario.algo_runs_timelimit
        # Instances
        num_train, num_test = [len([i for i in insts if i]) for insts in [scenario.train_insts, scenario.test_insts]]
        if num_train > 0 or num_test > 0:
            general['# instances (train/test)'] = "{} / {}".format(num_train, num_test)
        # Features
        num_feats = scenario.n_features if scenario.feature_dict else 0
        num_dup_feats = 0
        if scenario.feature_dict:
            dup_feats = DataFrame(scenario.feature_array)
            num_dup_feats = len(dup_feats[dup_feats.duplicated()])  # only contains train instances
        if num_feats > 0:
            general['# features (duplicates)'] = "{} ({})".format(num_feats, num_dup_feats)

        general['----------'] = '----------'

        combined_run = self.runscontainer.get_aggregated(False, False)[0]
        combined_stats = self._stats_for_run(combined_run.original_runhistory,
                                             combined_run.scenario,
                                             combined_run.incumbent)
        for k, v in combined_stats.items():
            general[k] = v

        return general

    def _runspec_dict(self, identify='parallel'):
        """
        identify-keyword specifies whether to use path or budget for name
        """
        if identify not in ['parallel', 'budget']:
            raise ValueError("illegal use of _runspec_dict")
        if (identify == 'budget' and len(self.runscontainer.get_budgets()) <= 1 and
            (self.runscontainer.get_budgets() is None or self.runscontainer.get_budgets()[0] == 0.0)):
            return False
        if (identify == 'parallel' and len(self.runscontainer.get_folders()) <= 1):
            return False

        runspec = OrderedDict()
        runs = self.runscontainer.get_aggregated(keep_folders=identify=='parallel',
                                                 keep_budgets=identify=='budget')

        for idx, run in enumerate(runs):
            if identify == 'budget' and len(set(run.reduced_to_budgets)) != 1:
                raise ValueError("Runs processed here should only have a single budget specified (%s)." %
                                 run.reduced_to_budgets)
            self.logger.debug("Path to folder for run no. {}: {}".format(idx, str(run.path_to_folder)))
            name = os.path.basename(run.path_to_folder) if identify == 'parallel' else str(run.reduced_to_budgets[0])
            runspec[name] = self._stats_for_run(run.original_runhistory,
                                                run.scenario,
                                                run.incumbent)
        return runspec

    def _stats_for_run(self, rh, scenario, incumbent):
        result = OrderedDict()

        all_configs = rh.get_all_configs()
        default = scenario.cs.get_default_configuration()

        # Runtime statistics
        all_ta_runtimes = [run_value.time for run_value in rh.data.values()]
        result['Total time spent evaluating configurations'] = "{:.2f} sec".format(np.sum(all_ta_runtimes))
        result['Average time per configuration (mean / std)'] = '{:5.2f} sec (± {:5.2f})'.format(np.mean(all_ta_runtimes),
                                                                                                 np.std(all_ta_runtimes))

        # Number of evaluations
        ta_evals = [len(rh.get_runs_for_config(c, only_max_observed_budget=True)) for c in all_configs]
        result['# evaluated configurations'] = len(all_configs)
        if not scenario.deterministic:
            result['# evaluations in total'] = np.sum(ta_evals)
            result['# evaluations for default/incumbent'] = "{}/{}".format(len(rh.get_runs_for_config(default, only_max_observed_budget=True)),
                                                                           len(rh.get_runs_for_config(incumbent, only_max_observed_budget=True)))
            result['# runs per configuration (min, mean and max)'] = "{}/{:.2f}/{}".format(
                            np.min(ta_evals), np.mean(ta_evals), np.max(ta_evals))
        # Info about configurations
        num_changed_params = len([p for p in scenario.cs.get_hyperparameter_names() if default[p] != incumbent[p]])
        result['# changed parameters (default to incumbent)'] = num_changed_params
        # Origins
        origins = [get_config_origin(c) for c in all_configs]
        origins = {o : origins.count(o) for o in set(origins)}
        if not (list(origins.keys()) == ["Unknown"]):
            result['Configuration origins'] = ", ".join(['{} : {}'.format(o, n) for o, n in origins.items()])

        return result

    def _configspace(self, cs):
        """ Return configspace in table-format """
        d = OrderedDict([("Parameter", []),
                         ("Type", []),
                         ("Range/Choices", []),
                         ("Default", [])]
                        )
        for hp in cs.get_hyperparameters():
            d["Parameter"].append(hp.name)
            d["Type"].append(type(hp).__name__)
            if isinstance(hp, NumericalHyperparameter):
                d["Range/Choices"].append("[{}, {}]{}".format(hp.lower, hp.upper, ' (log)' if hp.log else ''))
            elif isinstance(hp, CategoricalHyperparameter):
                d["Range/Choices"].append("{}".format(hp.choices))
            elif isinstance(hp, OrdinalHyperparameter):
                d["Range/Choices"].append("{}".format(hp.sequence))
            elif isinstance(hp, Constant):
                d["Range/Choices"].append("{}".format(hp.default_value))
            else:
                d["Range/Choices"].append("?")
            d["Default"].append(hp.default_value)
        return d
