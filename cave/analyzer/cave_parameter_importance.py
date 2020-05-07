import itertools
from collections import OrderedDict

import numpy as np
from bokeh.embed import components


from cave.analyzer.base_analyzer import BaseAnalyzer
from cave.plot.whisker_quantiles import whisker_quantiles
from cave.utils.hpbandster_helpers import format_budgets


class CaveParameterImportance(BaseAnalyzer):

    def __init__(self,
                 runscontainer,
                 ):
        """Calculate parameter-importance using the PIMP-package.
        """
        super().__init__(runscontainer)


    def parameter_importance(self, modus):
        """
        modus: str
            modus for parameter importance, from
            [forward-selection, ablation, fanova, lpi]
        """
        runs_by_budget = self.runscontainer.get_aggregated(keep_budgets=True, keep_folders=False)

        formatted_budgets = format_budgets(self.runscontainer.get_budgets(), allow_whitespace=True)

        self.result['Importances Per Parameter'] = {}
        result = self.result['Importances Per Parameter']
        for budget, run in zip(formatted_budgets.values(), runs_by_budget):
            self.logger.info("... parameter importance {} on {}".format(modus, run.get_identifier()))
            if not budget in result:
                result[budget] = OrderedDict()
            n_configs = len(run.original_runhistory.get_all_configs())
            n_params = len(run.scenario.cs.get_hyperparameters())
            if n_configs < n_params:
                result[budget] = {
                    'else' : "For this run there are only {} configs, "
                             "but {} parameters. No reliable parameter importance analysis "
                             "can be performed."}
                continue

            try:
                run.pimp.evaluate_scenario([modus], run.output_dir)
            except RuntimeError as e:
                err = "Encountered error '{}' for '{}' in '{}', (for fANOVA this can e.g. happen with too few " \
                      "data-points).".format(e, run.get_identifier(), modus)
                self.logger.info(err, exc_info=1)
                result[budget][modus + '_error'] = err
                continue
            individual_result = self.postprocess(run.pimp, run.output_dir)
            result[budget] = individual_result

            run.share_information['parameter_importance'][modus] = run.pimp.evaluator.evaluated_parameter_importance
            run.share_information['evaluators'][modus] = run.pimp.evaluator

        if self.runscontainer.analyzing_options['Parameter Importance'].getboolean('whisker_quantiles_plot'):
            if len(self.runscontainer.get_budgets()) <= 1 and len(self.runscontainer.get_folders()) <= 1:
                self.logger.info("The Whisker-Quantiles Plot for Parameter Importance makes only sense with multiple"
                                 "budgets and/or folders, but not with only one budget and one folder.")
                self.runscontainer.analyzing_options.set('Parameter Importance', 'whisker_quantiles_plot', 'False')
                self.importance_per_budget = None
                return

            hyperparameters = self.runscontainer.scenario.cs.get_hyperparameter_names()
            # Generate data - for each parallel folder and each budget, perform an importance-analysis
            importance_per_budget = OrderedDict()  # dict[budget][folder] -> (dict[param_name]->float)
            for budget in self.runscontainer.get_budgets():
                importance_per_budget[budget] = {hp : {} for hp in hyperparameters}
                for folder in self.runscontainer.get_folders():
                    cr = self.runscontainer.get_run(folder, budget)
                    try:
                        importance = cr.pimp.evaluate_scenario([modus],
                                                                cr.output_dir,
                                                                plot_pyplot=False,
                                                                plot_bokeh=False)[0][modus]['imp']
                    except RuntimeError as e:
                        importance = {}
                        err = "Encountered error '{}' for '{}' in '{}', (for fANOVA this can e.g. happen with too " \
                              "few data-points).".format(e, cr.get_identifier(), modus)
                        self.logger.debug(err, exc_info=1)
                        self.logger.error(err)

                    self.logger.debug("Importance for folder %s: %s", folder, importance)

                    for hp in hyperparameters:
                        importance_per_budget[budget][hp][folder] = importance.pop(hp, np.nan)
            self.importance_per_budget = importance_per_budget

    def plot_whiskers(self):
        if not self.importance_per_budget is None:
            return whisker_quantiles(self.importance_per_budget)


    def get_html(self, d=None, tooltip=None):
        if self.runscontainer.analyzing_options['Parameter Importance'].getboolean('whisker_quantiles_plot'):
            self.result['Whisker Plot'] = {'bokeh' : components(self.plot_whiskers()),
                                           'tooltip' : "Each dot is a parallel run (or folder) of the input data "
                                                       "and the whiskers are quartiles."}

        return super().get_html(d, tooltip)