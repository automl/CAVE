import os
from collections import OrderedDict

import numpy as np
from bokeh.embed import components
from bokeh.io import output_notebook
from bokeh.plotting import show
from pandas import DataFrame

from cave.analyzer.base_analyzer import BaseAnalyzer
from cave.utils.bokeh_routines import array_to_bokeh_table
from cave.utils.hpbandster_helpers import format_budgets


class PimpComparisonTable(BaseAnalyzer):
    """
    Parameters are initially sorted by pimp_sort_table_by. Only parameters with an importance greater than 5 in any
    of the methods are shown.  Note, that the values of the used methods are not directly comparable. For more
    information on the metrics, see respective tooltips."""

    def __init__(self,
                 runscontainer,
                 sort_table_by,
                 threshold=0.05):
        """Create a html-table over all evaluated parameter-importance-methods.
        Parameters are sorted after their average importance."""
        super().__init__(runscontainer)

        self.sort_table_by = sort_table_by
        self.threshold = threshold

    def get_name(self):
        return "Importance Table"

    def run(self):
        formatted_budgets = list(format_budgets(self.runscontainer.get_budgets(), allow_whitespace=True).values())
        for budget, run in zip(formatted_budgets, self.runscontainer.get_aggregated(keep_budgets=True, keep_folders=False)):
            self.result[budget] = self.plot(
                pimp=run.pimp,
                evaluators=list(run.share_information['evaluators'].values()),
                cs=self.runscontainer.scenario.cs,
                out_fn=os.path.join(run.output_dir, 'pimp.tex'),
            )

    def plot(self,
             pimp,
             evaluators,
             cs,
             out_fn,
             ):
        pimp.table_for_comparison(evaluators, out_fn, style='latex')
        self.logger.info('Creating pimp latex table at %s' % out_fn)

        parameters = [p.name for p in cs.get_hyperparameters()]
        index, values, columns = [], [], []
        columns = [e.name for e in evaluators]
        columns_lower = [c.lower() for c in columns]

        # SORT
        self.logger.debug("Sort pimp-table by %s" % self.sort_table_by)
        if self.sort_table_by == "average":
            # Sort parameters after average importance
            p_avg = {}
            for p in parameters:
                imps = [e.evaluated_parameter_importance[p] for e in evaluators if p in e.evaluated_parameter_importance]
                p_avg[p] = np.mean(imps) if imps else  0
            p_order = sorted(parameters, key=lambda p: p_avg[p], reverse=True)
        elif self.sort_table_by in columns_lower:
            def __get_key(p):
                imp = evaluators[columns_lower.index(self.sort_table_by)].evaluated_parameter_importance
                return imp[p] if p in imp else 0
            p_order = sorted(parameters, key=__get_key, reverse=True)
        else:
            raise ValueError("Trying to sort importance table after {}, which "
                             "was not evaluated.".format(self.sort_table_by))

        # PREPROCESS
        for p in p_order:
            values_for_p = [p]
            add_parameter = False  # Only add parameters where at least one evaluator shows importance > threshold
            for e in evaluators:
                if p in e.evaluated_parameter_importance:
                    # Check for threshold
                    value_to_add = e.evaluated_parameter_importance[p]
                    if value_to_add > self.threshold:
                        add_parameter = True
                    # All but forward-selection use values between 0 and 1
                    if e.name != 'Forward-Selection':
                        value_to_add = value_to_add * 100
                    # Create string and add uncertainty, if available
                    value_to_add = format(value_to_add, '05.2f')  # (leading zeros for sorting!)
                    if (hasattr(e, 'evaluated_parameter_importance_uncertainty') and
                        p in e.evaluated_parameter_importance_uncertainty):
                        value_to_add += ' +/- ' + format(e.evaluated_parameter_importance_uncertainty[p] * 100, '.2f')
                    values_for_p.append(value_to_add)
                else:
                    values_for_p.append('-')
            if add_parameter:
                values.append(values_for_p)

        # CREATE TABLE
        comp_table = DataFrame(values, columns=['Parameters'] + columns)
        sortable = {c : True for c in columns}
        width = {**{'Parameters' : 150}, **{c : 100 for c in columns}}

        bokeh_table = array_to_bokeh_table(comp_table, sortable=sortable, width=width, logger=self.logger)
        return {'bokeh' : bokeh_table}

    def get_html(self, d=None, tooltip=None):
        self.run()
        if len(self.result) == 1 and None in self.result:
            self.logger.debug("Detected None-key, abstracting away...")
            self.result = self.result[None]
        if d is not None:
            d[self.name] = OrderedDict()
            script, div = "", ""
        for b, t in self.result.items():
            s_, d_ = components(t) if b == 'bokeh' else components(t['bokeh'])
            script += s_
            div += d_
            if d is not None:
                if b == 'bokeh':
                    d[self.name] = {
                     "bokeh" : (s_, d_),
                     "tooltip" : self.__doc__,
                    }
                else:
                    d[self.name][b] = {
                        "bokeh" : (s_, d_),
                        "tooltip" : self.__doc__,
                    }
        return script, div

    def get_jupyter(self):
        self.run()
        output_notebook()
        for b, t in self.result.items():
            show(t['bokeh'])
