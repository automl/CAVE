import os
from collections import OrderedDict
import operator
import logging

from pandas import DataFrame
import numpy as np

from cave.analyzer.base_analyzer import BaseAnalyzer
from cave.html.html_helpers import figure_to_html

from bokeh.models import ColumnDataSource
from bokeh.models.widgets import DataTable, TableColumn
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

        # Only add parameters where at least one evaluator shows importance > threshold
        for p in p_order:
            values_for_p = []
            add_parameter = False
            for e in evaluators:
                if p in e.evaluated_parameter_importance:
                    value_to_add = e.evaluated_parameter_importance[p]
                    if e.name != 'Forward-Selection':
                        value_to_add = value_to_add * 100
                    value_to_add = format(value_to_add, '.2f')
                    if float(value_to_add) > threshold:
                        add_parameter = True
                    # Add uncertainty, if available
                    if (hasattr(e, 'evaluated_parameter_importance_uncertainty') and
                        p in e.evaluated_parameter_importance_uncertainty):
                        value_to_add += ' +/- ' + format(e.evaluated_parameter_importance_uncertainty[p] * 100, '.2f')
                    values_for_p.append(value_to_add)
                else:
                    values_for_p.append('-')
            if add_parameter:
                values.append(values_for_p)
                index.append(p)

        self.comp_table = DataFrame(values, columns=columns, index=index)
        self.bokeh_plot = self._pandaDF2bokehTable(self.comp_table)
        self.script, self.div = components(self.bokeh_plot)

    def _pandaDF2bokehTable(self, df):
        columns = list(df.columns.values)
        data = dict(df[columns])
        data["Parameters"] = df.index.tolist()
        source = ColumnDataSource(data)
        columns = [TableColumn(field='Parameters', title="Parameters", sortable=False, width=150)] + [
                   TableColumn(field=header, title=header, default_sort='descending', width=100) for header in columns
                  ]
        data_table = DataTable(source=source, columns=columns, height=20 + 30 * len(data["Parameters"]))
        return data_table

    def get_html(self, d=None, tooltip=None):
        table = self.comp_table.to_html()
        if d is not None:
            d["bokeh"] = self.script, self.div
        return self.script, self.div

    def get_jupyter(self):
        output_notebook()
        show(self.bokeh_plot)

