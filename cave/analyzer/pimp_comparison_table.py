import os
from collections import OrderedDict
import operator
import logging

from pandas import DataFrame
import numpy as np

from cave.analyzer.base_analyzer import BaseAnalyzer
from cave.html.html_helpers import figure_to_html
from cave.utils.bokeh_routines import array_to_bokeh_table

from bokeh.embed import components
from bokeh.plotting import show
from bokeh.io import output_notebook

class PimpComparisonTable(BaseAnalyzer):

    def __init__(self,
                 pimp,
                 evaluators,
                 sort_table_by,
                 cs,
                 out_fn,
                 threshold=0.05):
        """Create a html-table over all evaluated parameter-importance-methods.
        Parameters are sorted after their average importance."""
        self.logger = logging.getLogger(self.__module__ + '.' + self.__class__.__name__)
        self.sort_table_by = sort_table_by

        pimp.table_for_comparison(evaluators, out_fn, style='latex')
        self.logger.info('Creating pimp latex table at %s' % out_fn)

        parameters = [p.name for p in cs.get_hyperparameters()]
        index, values, columns = [], [], []
        columns = [e.name for e in evaluators]
        columns_lower = [c.lower() for c in columns]

        # SORT
        self.logger.debug("Sort pimp-table by %s" % sort_table_by)
        if sort_table_by == "average":
            # Sort parameters after average importance
            p_avg = {}
            for p in parameters:
                imps = [e.evaluated_parameter_importance[p] for e in evaluators if p in e.evaluated_parameter_importance]
                p_avg[p] = np.mean(imps) if imps else  0
            p_order = sorted(parameters, key=lambda p: p_avg[p], reverse=True)
        elif sort_table_by in columns_lower:
            def __get_key(p):
                imp = evaluators[columns_lower.index(sort_table_by)].evaluated_parameter_importance
                return imp[p] if p in imp else 0
            p_order = sorted(parameters, key=__get_key, reverse=True)
        else:
            raise ValueError("Trying to sort importance table after {}, which "
                             "was not evaluated.".format(sort_table_by))

        # PREPROCESS
        for p in p_order:
            values_for_p = [p]
            add_parameter = False  # Only add parameters where at least one evaluator shows importance > threshold
            for e in evaluators:
                if p in e.evaluated_parameter_importance:
                    # Check for threshold
                    value_to_add = e.evaluated_parameter_importance[p]
                    if value_to_add > threshold:
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
        self.comp_table = DataFrame(values, columns=['Parameters'] + columns)
        sortable = {c : True for c in columns}
        width = {**{'Parameters' : 150}, **{c : 100 for c in columns}}
        self.bokeh_plot = array_to_bokeh_table(self.comp_table, sortable=sortable, width=width, logger=self.logger)

        self.script, self.div = components(self.bokeh_plot)

    def get_html(self, d=None, tooltip=None):
        table = self.comp_table.to_html()
        if d is not None:
            d["bokeh"] = self.script, self.div
        return self.script, self.div

    def get_jupyter(self):
        output_notebook()
        show(self.bokeh_plot)

