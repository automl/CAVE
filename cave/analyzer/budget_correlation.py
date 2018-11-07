import os
import logging
from collections import OrderedDict

import numpy as np
from pandas import DataFrame
from typing import List
import scipy

from ConfigSpace.configuration_space import Configuration
from smac.runhistory.runhistory import RunHistory
from smac.scenario.scenario import Scenario

from bokeh.models import ColumnDataSource, CustomJS
from bokeh.models.widgets import DataTable, TableColumn
from bokeh.embed import components
from bokeh.plotting import show, figure
from bokeh.io import output_notebook
from bokeh.layouts import column
from bokeh.transform import jitter

from cave.analyzer.base_analyzer import BaseAnalyzer
from cave.utils.helpers import get_cost_dict_for_config, get_timeout, combine_runhistories
from cave.utils.statistical_tests import paired_permutation, paired_t_student
from cave.utils.timing import timing

class BudgetCorrelation(BaseAnalyzer):

    def __init__(self,
                 runs):
        """
        Parameters
        ----------
        incumbents: List[Configuration]
            incumbents per budget, assuming ascending order
        budget_names: List[str]
            budget-names as strings
        epm_rhs: List[RunHistory]
            estimated runhistories for budgets, same length and order as incumbents"""
        self.logger = logging.getLogger(self.__module__ + '.' + self.__class__.__name__)

        # To be set
        self.table = None
        self.dataframe = None
        self.create_table(runs)

    def create_table(self, runs):
        """Create table.

        Parameters
        ----------
        """
        table = []
        for b1 in runs:
            table.append([])
            for b2 in runs:
                configs = set(b1.combined_runhistory.get_all_configs()).intersection(set(b2.combined_runhistory.get_all_configs()))
                costs = list(zip(*[(b1.combined_runhistory.get_cost(c), b2.combined_runhistory.get_cost(c)) for c in configs]))
                self.logger.debug(costs)
                rho, p = scipy.stats.spearmanr(costs[0], costs[1])
                if runs.index(b2) < runs.index(b1):
                    table[-1].append("")
                else:
                    table[-1].append("{:.2f} ({} samples)".format(rho, len(costs[0])))

        budget_names = [os.path.basename(run.folder) for run in runs]
        df = DataFrame(data=table, columns=budget_names, index=budget_names)
        self.logger.debug(table)
        self.logger.debug(self.table)
        columns = list(df.columns.values)
        data = dict(df[columns])
        data["Budget"] = df.index.tolist()
        table_source = ColumnDataSource(data)
        columns = [TableColumn(field='Budget', title="Budget", sortable=False, width=20)] + [
                   TableColumn(field=header, title=header, default_sort='descending', width=10) for header in columns
                  ]
        bokeh_table = DataTable(source=table_source, columns=columns, row_headers=False, sortable=False,
                               height=20 + 30 * len(data["Budget"]))

        # Scatter
        all_configs = set([a for b in [run.original_runhistory.get_all_configs() for run in runs] for a in b])
        data = {os.path.basename(run.folder) : [run.original_runhistory.get_cost(c) if c in
            run.original_runhistory.get_all_configs() else
                                                None for c in all_configs] for run in runs}
        data['x'] = list(data.values())[0]
        data['y'] = list(data.values())[0]
        scatter_source = ColumnDataSource(data=data)
        # plot scatter

        p = figure(plot_width=400, plot_height=400,)

        # add a circle renderer with a size, color, and alpha
        p.circle(x='x', y='y',
                 #x=jitter('x', 0.1), y=jitter('y', 0.1),
                 source=scatter_source, size=5, color="navy", alpha=0.5)

        code = 'var budgets = ' + str(list(df.columns.values)) + ';'
        code += 'console.log(budgets);'
        code += """
        try {
            var grid = document.getElementsByClassName('grid-canvas')[0].children;
            var row = '';
            var col = '';
            for (var i=0,max=grid.length;i<max;i++){
                if (grid[i].outerHTML.includes('active')){
                    row=i;
                    for (var j=0,jmax=grid[i].children.length;j<jmax;j++){
                        if(grid[i].children[j].outerHTML.includes('active')){col=j}
                    }
                }
            }
            col = col - 1;
            console.log('row',row, budgets[row]);
            console.log('col',col, budgets[col]);
            cb_obj.selected['1d'].indices = [];

            if (row =>  0 && col > 0) {
              var new_x = scatter_source.data[budgets[row]].slice();
              var new_y = scatter_source.data[budgets[col]].slice();
              // Remove all pairs where one value is null
              while ((next_null = new_x.indexOf(null)) > -1) {
                new_x.splice(next_null, 1);
                new_y.splice(next_null, 1);
              }
              while ((next_null = new_y.indexOf(null)) > -1) {
                new_x.splice(next_null, 1);
                new_y.splice(next_null, 1);
              }
              scatter_source.data['x'] = new_x;
              scatter_source.data['y'] = new_y;
              scatter_source.change.emit();
              // Update axis-labels
              xaxis.attributes.axis_label = budgets[row];
              yaxis.attributes.axis_label = budgets[col];
            }
        } catch(err) {
            console.log(err.message);
            }
        """

        callback = CustomJS(args=dict(table_source=table_source,
                                      scatter_source=scatter_source,
                                      xaxis=p.xaxis[0],
                                      yaxis=p.yaxis[0],
                                      ), code=code)
        table_source.js_on_change('selected', callback)

        self.bokeh_plot = column(bokeh_table, p)
        self.script, self.div = components(self.bokeh_plot)

    def get_html(self, d=None, tooltip=None):
        if d is not None:
            d["bokeh"] = self.script, self.div
        return self.script, self.div

    def get_jupyter(self):
        output_notebook()
        show(self.bokeh_plot)


