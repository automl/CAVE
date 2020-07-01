import os
from typing import List, Union

import numpy as np
from ConfigSpace.configuration_space import Configuration
from smac.runhistory.runhistory import RunHistory

from cave.analyzer.base_analyzer import BaseAnalyzer
from cave.plot.scatter import plot_scatter_plot
from cave.utils.helpers import get_cost_dict_for_config, NotApplicable
from cave.utils.hpbandster_helpers import format_budgets


class PlotScatter(BaseAnalyzer):
    """
    Scatter plots show the costs of the default and optimized parameter configuration on each instance. Since this
    looses detailed information about the individual cost on each instance by looking at aggregated cost values in
    tables, scatter plots provide a more detailed picture. They provide insights whether overall performance
    improvements can be explained only by some outliers or whether they are due to improvements on the entire
    instance set. On the left side the training-data is scattered, on the right side the test-data is scattered.
    """
    def __init__(self,
                 runscontainer,
                 ):
        """
        Creates a scatterplot of the two configurations on the given set of instances.
        Saves plot to file.
        """
        super().__init__(runscontainer)

        formatted_budgets = format_budgets(self.runscontainer.get_budgets())
        for budget, run in zip(self.runscontainer.get_budgets(),
                               self.runscontainer.get_aggregated(keep_budgets=True, keep_folders=False)):
            self.result[formatted_budgets[budget]] = self._plot_scatter(
                    default=run.default,
                    incumbent=run.incumbent,
                    rh=run.epm_runhistory,
                    train=run.scenario.train_insts,
                    test=run.scenario.test_insts,
                    run_obj=run.scenario.run_obj,
                    cutoff=run.scenario.cutoff,
                    output_dir=run.output_dir,
            )

    def get_name(self):
        return "Scatter Plot"

    def _plot_scatter(self,
                      default: Configuration,
                      incumbent: Configuration,
                      rh: RunHistory,
                      train: List[str],
                      test: Union[List[str], None],
                      run_obj: str,
                      cutoff,
                      output_dir):
        """
        Parameters
        ----------
        default, incumbent: Configuration
            configurations to be compared
        rh: RunHistory
            runhistory to use for cost-estimations
        train[, test]: list(str)
            instance-names
        run_obj: str
            run-objective (time or quality)
        cutoff: float
            maximum runtime of ta
        output_dir: str
            output directory
        """
        out_fn_base = os.path.join(output_dir, 'scatter_')
        self.logger.info("... plotting scatter")

        metric = run_obj
        timeout = cutoff
        labels = ["default {}".format(run_obj), "incumbent {}".format(run_obj)]

        def_costs = get_cost_dict_for_config(rh, default).items()
        inc_costs = get_cost_dict_for_config(rh, incumbent).items()

        out_fns = []
        if len(train) <= 1 and len(test) <= 1:
            raise NotApplicable("No instances, so no scatter-plot.")
        for insts, name in [(train, 'train'), (test, 'test')]:
            if len(insts) <= 1:
                self.logger.debug("No %s instances, skipping scatter", name)
                continue
            default = np.array([v for k, v in def_costs if k in insts])
            incumbent = np.array([v for k, v in inc_costs if k in insts])
            min_val = min(min(default), min(incumbent))
            out_fn = out_fn_base + name + '.png'
            out_fns.append(plot_scatter_plot((default,), (incumbent,), labels, metric=metric,
                           min_val=min_val, max_val=timeout, out_fn=out_fn))
            self.logger.debug("Plotted scatter to %s", out_fn)
        return {'figure' : out_fns if len(out_fns) > 0 else None}
