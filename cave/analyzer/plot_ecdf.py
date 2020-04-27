import os
from typing import List

import numpy as np
from ConfigSpace.configuration_space import Configuration
from smac.runhistory.runhistory import RunHistory

from cave.analyzer.base_analyzer import BaseAnalyzer
from cave.plot.cdf import plot_cdf
from cave.utils.helpers import get_cost_dict_for_config, NotApplicable
from cave.utils.hpbandster_helpers import format_budgets


class PlotECDF(BaseAnalyzer):
    """
    Depicts cost distributions over the set of instances.  Since these are empirical distributions, the plots show
    step functions. These plots provide insights into how well configurations perform up to a certain threshold. For
    runtime scenarios this shows the probability of solving all instances from the set in a given timeframe. On the
    left side the training-data is scattered, on the right side the test-data is scattered.
    """
    def __init__(self,
                 runscontainer,
                 ):
        """
        Plot the cumulated distribution functions for given configurations,
        plots will share y-axis and if desired x-axis.
        Saves plot to file.
        """
        super().__init__(runscontainer)

        formatted_budgets = format_budgets(self.runscontainer.get_budgets())
        for budget, run in zip(self.runscontainer.get_budgets(),
                               self.runscontainer.get_aggregated(keep_budgets=True, keep_folders=False)):
            self.result[formatted_budgets[budget]] = self._plot_ecdf(
                run.default,
                run.incumbent,
                run.epm_runhistory,
                run.scenario.train_insts,
                run.scenario.test_insts,
                run.scenario.cutoff,
                run.output_dir)

    def get_name(self):
        return "empirical Cumulative Distribution Function (eCDF)"

    def _plot_ecdf(self,
                   default: Configuration,
                   incumbent: Configuration,
                   rh: RunHistory,
                   train: List[str],
                   test: List[str],
                   cutoff,
                   output_dir: str):
        """
        Parameters
        ----------
        default, incumbent: Configuration
            configurations to be compared
        rh: RunHistory
            runhistory to use for cost-estimations
        train, test: List[str]
            lists with corresponding instances
        cutoff: Union[None, int]
            cutoff for target algorithms, if set
        output_dir: str
            directory to save plots in
        """
        out_fn_base = os.path.join(output_dir, 'cdf')
        self.logger.info("... plotting eCDF")

        def prepare_data(x_data):
            """ Helper function to keep things easy, generates y_data and manages x_data-timeouts """
            x_data = sorted(x_data)
            y_data = np.array(range(len(x_data))) / (len(x_data) - 1)
            for idx in range(len(x_data)):
                if (cutoff is not None) and (x_data[idx] >= cutoff):
                    x_data[idx] = cutoff
                    y_data[idx] = y_data[idx - 1]
            return (x_data, y_data)

        # Generate y_data
        def_costs = get_cost_dict_for_config(rh, default).items()
        inc_costs = get_cost_dict_for_config(rh, incumbent).items()

        output_fns = []
        if len(train) <= 1 and len(test) <= 1:
            raise NotApplicable("No instances, so no eCDF-plot.")
        for insts, name in [(train, 'train'), (test, 'test')]:
            if len(insts) <= 1:
                self.logger.debug("No %s instances, skipping cdf", name)
                continue
            data = [prepare_data(np.array([v for k, v in costs if k in insts])) for costs in [def_costs, inc_costs]]
            x, y = (data[0][0], data[1][0]), (data[0][1], data[1][1])
            labels = ['default ' + name, 'incumbent ' + name]
            out_fn = out_fn_base + '_{}.png'.format(name)
            output_fns.append(plot_cdf(x, y, labels, timeout=cutoff,
                                       out_fn=out_fn))
            self.logger.debug("Plotted eCDF to %s", out_fn)
        return {'figure' : output_fns if len(output_fns) > 0 else None}
