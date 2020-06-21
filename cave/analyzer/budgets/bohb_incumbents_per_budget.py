from typing import List

import numpy as np
from ConfigSpace.configuration_space import Configuration
from pandas import DataFrame
from smac.runhistory.runhistory import RunHistory

from cave.analyzer.base_analyzer import BaseAnalyzer
from cave.utils.helpers import get_cost_dict_for_config
from cave.utils.hpbandster_helpers import format_budgets


class BohbIncumbentsPerBudget(BaseAnalyzer):
    # todo rename (not BOHB-specific)
    """
    Show the incumbents for each budget (i.e. the best configuration by kernel-estimation using data from that
    budget).
    """

    def __init__(self,
                 runscontainer,
                 ):
        super().__init__(runscontainer)

        runs = runscontainer.get_aggregated(keep_budgets=True, keep_folders=False)
        incumbents = [r.incumbent for r in runs]
        budget_names = [f for b, f in format_budgets(runscontainer.get_budgets(), allow_whitespace=True).items()]
        epm_rhs = [r.epm_runhistory for r in runs]

        self.create_table(incumbents, budget_names, epm_rhs)

    def get_name(self):
        return "Incumbents Over Budgets"

    def create_table(self, incumbents, budget_names, epm_rhs):
        """Create table.

        Parameters
        ----------
        incumbents: List[Configuration]
            incumbents per budget, assuming ascending order
        budget_names: List[str]
            budget-names as strings
        epm_rhs: List[RunHistory]
            estimated runhistories for budgets, same length and order as incumbents
        """
        self.logger.info("... create performance table")
        if not (len(incumbents) == len(epm_rhs) and len(incumbents) == len(budget_names)):
            raise ValueError("Number of incumbents must equal number of names and runhistories")

        dec_place = 3

        # Get costs
        costs = []
        for inc, epm_rh in zip(incumbents, epm_rhs):
            cost_dict_inc = get_cost_dict_for_config(epm_rh, inc)
            costs.append(np.mean([float(v) for v in cost_dict_inc.values()]))

        keys = [k for k in incumbents[0].keys() if any([inc[k] for inc in incumbents])]
        values = []
        for inc, c in zip(incumbents, costs):
            new_values = [inc[k] if inc[k] is not None else "inactive" for k in keys]
            new_values.append(str(round(c, dec_place)))
            values.append(new_values)

        keys.append('Cost')
        table = list(zip(keys, *values))
        keys, table = [k[0] for k in table], [k[1:] for k in table]
        df = DataFrame(data=table, columns=budget_names, index=keys)
        self.result['table'] = df.to_html()
