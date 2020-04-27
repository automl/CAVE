import itertools
from collections import OrderedDict

import numpy as np
from bokeh.embed import components
from bokeh.models import ColumnDataSource, Whisker
from bokeh.models import FactorRange, Range1d
from bokeh.palettes import d3
from bokeh.plotting import figure
from bokeh.transform import dodge

from cave.analyzer.base_analyzer import BaseAnalyzer
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
        runs_by_budget = self.runscontainer.get_aggregated(keep_budgets=True,
                                                           keep_folders=False)

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
                        err = "Encountered error '{}' for '{}' in '{}', (for fANOVA this can e.g. happen with too few " \
                              "data-points).".format(e, cr.get_identifier(), modus)
                        self.logger.debug(err, exc_info=1)
                        self.logger.error(err)

                    self.logger.debug("Importance for folder %s: %s", folder, importance)

                    for hp in hyperparameters:
                        importance_per_budget[budget][hp][folder] = importance.pop(hp, np.nan)
            self.importance_per_budget = importance_per_budget


    def plot_whiskers(self):
        importance_per_budget = self.importance_per_budget
        # Bokeh plot
        colors = itertools.cycle(d3['Category10'][len(importance_per_budget)])
        hyperparameters = self.runscontainer.scenario.cs.get_hyperparameter_names()

        whiskers_data = {}
        for b in importance_per_budget.keys():
            whiskers_data.update({'base_' + str(b): [], 'lower_' + str(b): [], 'upper_' + str(b): []})
        self.logger.debug("Importance per budget: %s", str(importance_per_budget))
        # Generate whiskers data
        for (b, imp_dict) in importance_per_budget.items():
            for p, imp in imp_dict.items():
                mean = np.nanmean(np.array(list(imp.values())))
                std = np.nanstd(np.array(list(imp.values())))
                if not np.isnan(mean) and not np.isnan(std):
                    whiskers_data['lower_' + str(b)].append(mean - std)
                    whiskers_data['upper_' + str(b)].append(mean + std)
                    whiskers_data['base_' + str(b)].append(p)
        whiskers_datasource = ColumnDataSource(whiskers_data)

        self.logger.debug("Hyperparameters: %s", str(hyperparameters))
        plot = figure(x_range=FactorRange(factors=hyperparameters, bounds='auto'), y_range=Range1d(0, 1, bounds='auto'),
                      plot_width=600, plot_height=300,
                      title="folders per budget with quartile ranges")

        dodgies = np.linspace(-0.25, 0.25, len(importance_per_budget))
        # Plot
        for (b, imp_dict), d, color in zip(importance_per_budget.items(), dodgies, colors):
            for p, imp in imp_dict.items():
                self.logger.debug('imp: %s', str(imp))
                for i in imp.values():
                    if np.isnan(i):
                        continue
                    self.logger.debug("%s, %s, %s, %s", b, d, p, i)
                    plot.circle(x=[(p, d)], y=[i], color=color, fill_alpha=0.4, legend="Budget %s" % str(b))

            if not 'base_' + str(b) in whiskers_data:
                continue
            plot.add_layout(Whisker(source=whiskers_datasource,
                                    base=dodge('base_' + str(b), d, plot.x_range),
                                    lower='lower_' + str(b),
                                    upper='upper_' + str(b),
                                    line_color=color))
        #self.result['Whisker Plot'] = {'bokeh' : plot}
        return plot

    def get_html(self, d=None, tooltip=None):
        if self.runscontainer.analyzing_options['Parameter Importance'].getboolean('whisker_quantiles_plot'):
            self.result['Whisker Plot'] = {'bokeh' : components(self.plot_whiskers())}
        return super().get_html(d, tooltip)