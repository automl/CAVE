import os
from collections import namedtuple

from bokeh.embed import components
from bokeh.io import output_notebook
from bokeh.plotting import show

from cave.analyzer.apt.base_apt import BaseAPT
from cave.reader.runs_container import RunsContainer
from cave.utils.hpbandster_helpers import format_budgets

Line = namedtuple('Line', ['name', 'time', 'mean', 'upper', 'lower', 'config'])

class LossCurves(BaseAPT):
    """
    Only works with AutoPyTorch-instance.
    Visualize loss-curves of multiple neural networks for comparison in interactive plot.
    """

    def __init__(self,
                 runscontainer: RunsContainer,
                 incumbent_trajectory: str=None,
                 ):
        """
        """
        super().__init__(runscontainer,
                         incumbent_trajectory=incumbent_trajectory,
                         )

        self.rng = self.runscontainer.get_rng()

        self.scenario = self.runscontainer.scenario
        self.output_dir = os.path.join(self.runscontainer.output_dir, "tensorboard")
        self.rh = self.runscontainer.get_aggregated(False, False)[0].validated_runhistory
        # Run-specific / budget specific infos
        if len(self.runscontainer.get_budgets()) > 1:
            self.runs = self.runscontainer.get_aggregated(keep_folders=False, keep_budgets=True)
        else:
            self.runs = self.runscontainer.get_aggregated(keep_folders=True, keep_budgets=False)

        self.formatted_budgets = format_budgets(self.runscontainer.get_budgets())

        # Will be set during execution:
        self.plots = []                     # List with paths to '.png's

    def get_name(self):
        return "Loss Curves"

    def plot(self):
        """
        Plot performance over time, using all trajectory entries.
        max_time denotes max(wallclock_limit, highest recorded time).
        """
        #TODO Read in Tensorboard information
        #TODO interactive loss-plots
        raise NotImplementedError()

    def get_plots(self):
        return self.plots


    def get_html(self, d=None, tooltip=None):
        script, div = components(self.plot())
        if d is not None:
            d[self.name] = {
                "bokeh" : (script, div),
                "tooltip" : self.__doc__,
            }
        return script, div

    def get_jupyter(self):
        output_notebook()
        show(self.plot())
