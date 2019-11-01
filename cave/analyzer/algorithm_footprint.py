import os

from bokeh.embed import components
from bokeh.io import output_notebook
from bokeh.plotting import show

from cave.analyzer.base_analyzer import BaseAnalyzer
from cave.plot.algorithm_footprint import AlgorithmFootprintPlotter
from cave.utils.helpers import NotApplicable
from cave.utils.timing import timing


class AlgorithmFootprint(BaseAnalyzer):
    """
    The instance features are projected into a two/three dimensional space using principal component analysis (PCA)
    and the footprint of each algorithm is plotted, i.e., on which instances the default or the optimized
    configuration performs well. In contrast to the other analysis methods in this section, these plots allow
    insights into which of the two configurations performs well on specific types or clusters of instances. Inspired
    by Smith-Miles.
    """

    @timing
    def __init__(self,
                 runscontainer,
                 density=200,
                 purity=0.95):
        """
        Parameters
        ----------
        runscontainer: RunsContainer
            contains all important information about the configurator runs
        """
        super().__init__(runscontainer)

        # Aggregated run over current runscontainer
        self.logger.info("Note: Algorithm Footprint does not support budgets / fidelities yet.")
        agg_run = self.runscontainer.get_aggregated(False, False)[0]

        algorithms = [(agg_run.default, "default"),
                      (agg_run.incumbent, "incumbent")]
        epm_rh = agg_run.epm_runhistory
        train = agg_run.scenario.train_insts
        test = agg_run.scenario.test_insts
        features = agg_run.scenario.feature_dict
        cutoff = agg_run.scenario.cutoff
        output_dir = agg_run.output_dir
        rng = agg_run.rng

        # filter instance features
        self.logger.debug("Features: " + str(features))
        train_feats = {k: v for k, v in features.items() if k in train}
        test_feats = {k: v for k, v in features.items() if k in test}
        if not (train_feats or test_feats):
            self.logger.warning("No instance-features could be detected. "
                                "No algorithm footprints available.")
            raise NotApplicable("Could not detect any instances.")

        self.logger.info("... algorithm footprints for: {}".format(",".join([a[1] for a in algorithms])))

        try:
            self.footprint = AlgorithmFootprintPlotter(epm_rh,
                                                   train_feats, test_feats,
                                                   algorithms,
                                                   cutoff,
                                                   output_dir,
                                                   rng=rng)
        except ValueError as err:
            self.logger.debug(err, exc_info=1)
            self.error = str(err)

    def get_name(self):
        return "Algorithm Footprint"

    def _plot(self):
        # Plot footprints
        bokeh_plot = self.footprint.plot_interactive_footprint()
        self.plots3d = self.footprint.plot3d()
        return bokeh_plot

    def get_jupyter(self):
        bokeh_plot = self._plot()
        output_notebook()
        show(bokeh_plot)

    def get_html(self, d=None, tooltip=None):

        if self.error:
            if d is not None:
                d["Algorithm Footprint"] = {"else": self.error}
            return self.error
        else:
            bokeh_components = components(self._plot())
            if d is not None:
                d["Algorithm Footprint"] = {"tooltip" : self.__doc__}
                # Interactive bokeh-plot
                d["Algorithm Footprint"]["Interactive Algorithm Footprint"] = {"bokeh" : bokeh_components}
                for plots in self.plots3d:
                    header = os.path.splitext(os.path.split(plots[0])[1])[0][10:-2]
                    header = header[0].upper() + header[1:].replace('_', ' ')
                    d["Algorithm Footprint"][header] = {"figure_x2": plots}
            return bokeh_components

    def get_plots(self):
        all_plots = []
        for plots in self.plots3d:
            all_plots.extend(plots)
        return all_plots
