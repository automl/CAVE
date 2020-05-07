import os

import numpy as np
from bokeh.embed import components
from bokeh.io import output_notebook
from bokeh.plotting import show

from cave.analyzer.base_analyzer import BaseAnalyzer
from cave.plot.configurator_footprint import ConfiguratorFootprintPlotter


class ConfiguratorFootprint(BaseAnalyzer):
    """
    Analysis of the iteratively sampled configurations during the optimization procedure.  Multi-dimensional scaling
    (MDS) is used to reduce dimensionality of the search space and plot the distribution of evaluated
    configurations. The larger the dot, the more often the configuration was evaluated on instances from the set.
    Configurations that were incumbents at least once during optimization are marked as red squares.  Configurations
    acquired through local search are marked with a 'x'.  The downward triangle denotes the final incumbent, whereas
    the orange upward triangle denotes the default configuration.  The heatmap and the colorbar correspond to the
    predicted performance in that part of the search space.
    """

    def __init__(self,
                 runscontainer,
                 max_configurations_to_plot=None,
                 time_slider=None,
                 number_quantiles=None,
                 timeslider_log: bool=None,
                 ):
        """Plot the visualization of configurations, highlighting the
        incumbents. Using original rh, so the explored configspace can be
        estimated.

        Parameters
        ----------
        runscontainer: RunsContainer
            contains all important information about the configurator runs
        max_configurations_to_plot: int
            maximum number of data-points to plot
        time_slider: bool
            whether or not to have a time_slider-widget on cfp-plot
            INCREASES FILE-SIZE DRAMATICALLY
        number_quantiles: int
            if use_timeslider is not off, defines the number of quantiles for the
            slider/ number of static pictures
        timeslider_log: bool
            whether to use a logarithmic scale for the timeslider/quantiles

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
        super().__init__(runscontainer,
                         max_configurations_to_plot=max_configurations_to_plot,
                         time_slider=time_slider,
                         number_quantiles=number_quantiles,
                         timeslider_log=timeslider_log)

        self.logger.info("... visualizing explored configspace (this may take "
                         "a long time, if there is a lot of data - deactive with --no_configurator_footprint)")
        self.output_dir = self.runscontainer.output_dir
        self.scenario = self.runscontainer.scenario
        # Run-specific / budget specific infos
        if len(self.runscontainer.get_budgets()) > 1:
            self.runs = self.runscontainer.get_aggregated(keep_folders=False, keep_budgets=True)
            rh_labels = ["Budget " + str(r.reduced_to_budgets[0]) for r in self.runs]
        else:
            self.runs = self.runscontainer.get_aggregated(keep_folders=True, keep_budgets=False)
            rh_labels = [os.path.basename(r.path_to_folder).replace('_', ' ') for r in self.runs]
        self.logger.debug("Analyzing runs: {}".format([r.get_identifier() for r in self.runs]))

        self.max_confs = self.options.getint('max_configurations_to_plot')
        self.use_timeslider = self.options.getboolean('time_slider')
        self.num_quantiles = self.options.getint('number_quantiles')
        self.timeslider_log = self.options.getboolean('timeslider_log')

        incumbents = {r.trajectory[-1]['incumbent'] : r.trajectory[-1]['cost'] for r in self.runs}
        self.final_incumbent = min(incumbents, key=incumbents.get)

        if self.scenario.feature_array is None:
            self.scenario.feature_array = np.array([[]])

        self.cfp = ConfiguratorFootprintPlotter(
                       scenario=self.scenario,
                       rhs=[r.original_runhistory for r in self.runs],
                       incs=[list(incumbents.keys())],
                       final_incumbent=self.final_incumbent,
                       rh_labels=rh_labels,
                       max_plot=self.max_confs,
                       use_timeslider=self.use_timeslider and self.num_quantiles > 1,
                       num_quantiles=self.num_quantiles,
                       timeslider_log=self.timeslider_log,
                       output_dir=self.output_dir)

    def get_name(self):
        return "Configurator Footprint"

    def plot(self):
        try:
            res = self.cfp.run()
        except MemoryError as err:
            self.logger.exception(err)
            raise MemoryError("Memory Error occured in configurator footprint. "
                              "You may want to reduce the number of plotted "
                              "configs (using the '--cfp_max_plot'-argument)")

        bokeh_plot, self.cfp_paths = res
        return bokeh_plot

    def get_jupyter(self):
        bokeh_plot = self.plot()
        output_notebook()
        show(bokeh_plot)

    def get_html(self, d=None, tooltip=None):
        bokeh_components = components(self.plot())
        if d is not None:
            if self.num_quantiles == 1 or self.use_timeslider:  # No need for "Static" with one plot / time slider activated
                d[self.name] = {
                    "bokeh" : bokeh_components,
                    "tooltip": self.__doc__,
                }
            else:
                d[self.name] = {
                    "tooltip": self.__doc__,
                    "Interactive" : {"bokeh": (bokeh_components)},
                }
                if all([True for p in self.cfp_paths if os.path.exists(p)]):  # If the plots were actually generated
                    d[self.name]["Static"] = {"figure": self.cfp_paths}
                else:
                    d[self.name]["Static"] = {
                            "else": "This plot is missing. Maybe it was not generated? "
                                    "Check if you installed selenium and phantomjs "
                                    "correctly to activate bokeh-exports. "
                                    "(https://automl.github.io/CAVE/stable/faq.html)"}
        return bokeh_components

    def get_plots(self):
        return self.cfp_paths
