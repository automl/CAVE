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

from bokeh.models import ColumnDataSource
from bokeh.models.widgets import DataTable, TableColumn
from bokeh.embed import components
from bokeh.plotting import show
from bokeh.io import output_notebook

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
        self.logger.info("... create performance table")

        table = []
        for b1 in runs:
            table.append([])
            for b2 in runs:
                configs = set(b1.combined_runhistory.get_all_configs()).intersection(set(b2.combined_runhistory.get_all_configs()))
                costs = list(zip(*[(b1.combined_runhistory.get_cost(c), b2.combined_runhistory.get_cost(c)) for c in configs]))
                self.logger.debug(costs)
                rho, p = scipy.stats.spearmanr(costs[0], costs[1])
                table[-1].append("{:.2f}/{:.2f}".format(rho, p))

        budget_names = [os.path.basename(run.folder) for run in runs]
        self.table = DataFrame(data=table, columns=budget_names, index=budget_names)
        self.logger.debug(table)
        self.logger.debug(self.table)
        self.bokeh_plot = self._pandaDF2bokehTable(self.table)
        self.script, self.div = components(self.bokeh_plot)

    def _pandaDF2bokehTable(self, df):
        columns = list(df.columns.values)
        data = dict(df[columns])
        data["Budget"] = df.index.tolist()
        source = ColumnDataSource(data)
        columns = [TableColumn(field='Budget', title="Budget", sortable=False, width=20)] + [
                   TableColumn(field=header, title=header, default_sort='descending', width=10) for header in columns
                  ]
        data_table = DataTable(source=source, columns=columns, row_headers=False, sortable=False,
                               height=20 + 30 * len(data["Budget"]))
        return data_table

    def get_html(self, d=None, tooltip=None):
        if d is not None:
            d["bokeh"] = self.script, self.div
        return self.script, self.div

    def get_jupyter(self):
        output_notebook()
        show(self.bokeh_plot)


