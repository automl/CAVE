import os
from collections import OrderedDict
import logging

import numpy as np
from bokeh.embed import components

from cave.analyzer.base_analyzer import BaseAnalyzer
from cave.plot.configurator_footprint import ConfiguratorFootprintPlotter

from bokeh.plotting import show
from bokeh.io import output_notebook

class ConfiguratorFootprint(BaseAnalyzer):

    def __init__(self,
                 scenario,
                 runs,
                 runhistory,
                 output_dir,
                 max_confs=1000,
                 time_slider=False,
                 num_quantiles=10):
        """Plot the visualization of configurations, highlighting the
        incumbents. Using original rh, so the explored configspace can be
        estimated.

        Parameters
        ----------
        scenario: Scenario
            deepcopy of scenario-object
        runs: List[ConfiguratorRun]
            holding information about original runhistories, trajectories, incumbents, etc.
        runhistory: RunHistory
            with maximum number of real (not estimated) runs to train best-possible epm
        max_confs: int
            maximum number of data-points to plot
        time_slider: bool
            whether or not to have a time_slider-widget on cfp-plot
            INCREASES FILE-SIZE DRAMATICALLY
        num_quantiles: int
            if time_slider is not off, defines the number of quantiles for the
            slider/ number of static pictures

        Returns
        -------
        script: str
            script part of bokeh plot
        div: str
            div part of bokeh plot
        over_time_paths: List[str]
            list with paths to the different quantiled timesteps of the
            configurator run (for static evaluation)
        """
        self.logger = logging.getLogger(self.__module__ + '.' + self.__class__.__name__)
        self.logger.info("... visualizing explored configspace (this may take "
                         "a long time, if there is a lot of data - deactive with --no_configurator_footprint)")

        self.scenario = scenario
        self.runs = runs
        self.runhistory = runhistory
        self.output_dir = output_dir
        self.max_confs = max_confs
        self.time_slider = time_slider
        self.num_quantiles = num_quantiles

        if scenario.feature_array is None:
            scenario.feature_array = np.array([[]])

        # Sort runhistories and incs wrt cost
        incumbents = list(map(lambda x: x['incumbent'], runs[0].traj))
        assert(incumbents[-1] == runs[0].traj[-1]['incumbent'])

        cfp = ConfiguratorFootprintPlotter(
                       scenario=scenario,
                       rh=runhistory,
                       incs=incumbents,
                       max_plot=max_confs,
                       time_slider=time_slider and num_quantiles > 1,
                       num_quantiles=num_quantiles,
                       output_dir=output_dir)
        try:
            res = cfp.run()
        except MemoryError as err:
            self.logger.exception(err)
            raise MemoryError("Memory Error occured in configurator footprint. "
                              "You may want to reduce the number of plotted "
                              "configs (using the '--cfp_max_plot'-argument)")

        self.bokeh_plot, self.cfp_paths = res
        self.script, self.div = components(self.bokeh_plot)

    def get_jupyter(self):
        output_notebook()
        if not self.bokeh_plot:
            self.bokeh_plot = self._plot()
        show(self.bokeh_plot)

    def get_html(self, d=None):
        bokeh_components = self.script, self.div
        if d is not None:
            if self.num_quantiles == 1:  # Only one plot, no need for "Static"-field
                d["Configurator Footprint"] = {"bokeh": (bokeh_components)}
            else:
                d["Configurator Footprint"] = OrderedDict()
                d["Configurator Footprint"]["Interactive"] = {"bokeh": (bokeh_components)}
                if all([True for p in self.cfp_paths if os.path.exists(p)]):  # If the plots were actually generated
                    d["Configurator Footprint"]["Static"] = {"figure": self.cfp_paths}
                else:
                    d["Configurator Footprint"]["Static"] = {
                            "else": "This plot is missing. Maybe it was not generated? "
                                    "Check if you installed selenium and phantomjs "
                                    "correctly to activate bokeh-exports. "
                                    "(https://automl.github.io/CAVE/stable/faq.html)"}
        return bokeh_components

    def get_plots(self):
        return self.cfp_paths
