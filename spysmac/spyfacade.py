import os
import logging
from collections import OrderedDict
from contextlib import contextmanager
import typing
import json
import copy

import numpy as np
from pandas import DataFrame

from smac.configspace import Configuration
from smac.epm.rf_with_instances import RandomForestWithInstances
from smac.optimizer.objective import average_cost
from smac.runhistory.runhistory import RunKey, RunValue, RunHistory
from smac.runhistory.runhistory2epm import RunHistory2EPM4Cost
from smac.scenario.scenario import Scenario
from smac.utils.io.traj_logging import TrajLogger
from smac.utils.io.input_reader import InputReader
from smac.utils.validate import Validator

from pimp.importance.importance import Importance

from spysmac.html.html_builder import HTMLBuilder
from spysmac.plot.plotter import Plotter
from spysmac.smacrun import SMACrun
from spysmac.analyzer import Analyzer
from spysmac.utils.helpers import get_cost_dict_for_config

from spysmac.asapy.feature_analysis import FeatureAnalysis
from spysmac.plot.algorithm_footprint import AlgorithmFootprint

__author__ = "Joshua Marben"
__copyright__ = "Copyright 2017, ML4AAD"
__license__ = "3-clause BSD"
__maintainer__ = "Joshua Marben"
__email__ = "joshua.marben@neptun.uni-freiburg.de"

@contextmanager
def changedir(newdir):
    """ Helper function to change directory, for example to create a scenario
    from file, where paths to the instance- and feature-files are relative to
    the original SMAC-execution-directory. Same with target algorithms that need
    be executed for validation. """
    olddir = os.getcwd()
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(olddir)

class SpySMAC(object):
    """
    """

    def __init__(self, folders: typing.List[str], output: str,
                 ta_exec_dir: str='.', missing_data_method: str='epm'):
        """
        Initialize SpySMAC facade to handle analyzing, plotting and building the
        report-page easily. During initialization, the analysis-infrastructure
        is built and the data is validated, meaning the overall best
        incumbent is found and default+incumbent are evaluated for all
        instances, by default using an EPM.
        The analyze()-method performs an analysis and outputs a report.html.

        Arguments
        ---------
        folders: list<strings>
            paths to relevant SMAC runs
        output: string
            output for spysmac to write results (figures + report)
        ta_exec_dir: string
            execution directory for target algorithm
        missing_data_method: string
            from [validation, epm], how to estimate missing runs
        """
        self.logger = logging.getLogger("spysmac.spyfacade")
        self.ta_exec_dir = ta_exec_dir

        # Create output if necessary
        self.output = output
        self.logger.info("Writing to %s", self.output)
        if not os.path.exists(output):
            self.logger.debug("Output-dir %s does not exist, creating", self.output)
            os.makedirs(output)

        # Global runhistory combines all actual runs of individual SMAC-runs
        # We save the combined (unvalidated) runhistory to disk, so we can use it later on.
        # We keep the validated runhistory (with as many runs as possible) in
        # memory. The distinction is made to avoid using runs that are
        # only estimated using an EPM for further EPMs or to handle runs
        # validated on different hardware (depending on validation-method).
        self.original_rh = RunHistory(average_cost)
        self.validated_rh = RunHistory(average_cost)

        # Save all relevant SMAC-runs in a list
        self.runs = []
        for folder in folders:
            try:
                self.logger.debug("Collecting data from %s.", folder)
                self.runs.append(SMACrun(folder, ta_exec_dir))
            except Exception as err:
                self.logger.warning("Folder %s could not be loaded, failed "
                                    "with error message: %s", folder, err)
                continue
        if not len(self.runs):
            raise ValueError("None of the specified SMAC-folders could be loaded.")

        # Use scenario of first run for general purposes (expecting they are all the same anyway!)
        self.scenario = self.runs[0].solver.scenario

        # Update global runhistory with all available runhistories
        self.logger.debug("Update original rh with all available rhs!")
        runhistory_fns = [os.path.join(run.folder, "runhistory.json") for run in self.runs]
        for rh_file in runhistory_fns:
            self.original_rh.update_from_json(rh_file, self.scenario.cs)
        self.logger.debug('Combined number of Runhistory data points: %d. '
                          '# Configurations: %d. # Runhistories: %d',
                          len(runhistory_fns), len(self.original_rh.data),
                          len(self.original_rh.get_all_configs()))
        self.original_rh.save_json(os.path.join(self.output, "combined_rh.json"))

        # Estimate all missing costs using validation or EPM
        self.complete_data(method=missing_data_method)
        self.best_run = min(self.runs, key=lambda run:
                self.validated_rh.get_cost(run.solver.incumbent))

        self.default = self.scenario.cs.get_default_configuration()
        self.incumbent = self.best_run.solver.incumbent

        # Following variable determines whether a distinction is made
        # between train and test-instances (e.g. in plotting)
        self.train_test = bool(self.scenario.train_insts != [None] and
                               self.scenario.test_insts != [None])

        self.analyzer = Analyzer(self.original_rh, self.validated_rh,
                                 self.default, self.incumbent, self.train_test,
                                 self.scenario, self.output)

        self.website = OrderedDict([])

    def complete_data(self, method="epm"):
        """Complete missing data of runs to be analyzed. Either using validation
        or EPM.
        """
        with changedir(self.ta_exec_dir):
            self.logger.info("Completing data using %s.", method)

            path_for_validated_rhs = os.path.join(self.output, "validated_rhs")
            for run in self.runs:
                # out = os.path.join(path_for_validated_rhs, "rh_"+run.folder)
                out = ""
                validator = Validator(run.scen, run.traj, out)

                if method == "validation":
                    # TODO determine # repetitions
                    new_rh = validator.validate('def+inc', 'train+test', 1, -1,
                                                runhistory=self.original_rh)
                elif method == "epm":
                    new_rh = validator.validate_epm('def+inc', 'train+test', 1,
                                                    runhistory=self.original_rh)
                else:
                    raise ValueError("Missing data method illegal (%s)",
                                     method)
                self.validated_rh.update(new_rh)

    def analyze(self,
                performance=True, cdf=True, scatter=True, confviz=True,
                param_importance=['forward_selection', 'ablation', 'fanova'],
                feature_analysis=["box_violin", "correlation",
                    "feat_importance", "clustering", "feature_cdf"],
                parallel_coordinates=True, cost_over_time=True,
                algo_footprint=True):
        """Analyze the available data and build HTML-webpage as dict.
        Save webpage in 'self.output/SpySMAC/report.html'.
        Analyzing is performed with the analyzer-instance that is initialized in
        the __init__, same with plotting and plotter-instance..

        Parameters
        ----------
        par10: bool
            whether to calculate par10-values
        cdf: bool
            whether to plot cdf
        scatter: bool
            whether to plot scatter
        forward_selection: bool
            whether to apply forward selection
        ablation: bool
            whether to apply ablation
        fanova: bool
            whether to apply fanova
        """
        builder = HTMLBuilder(self.output, "SpySMAC")
        # Check arguments
        for p in param_importance:
            if p not in ['forward_selection', 'ablation', 'fanova']:
                raise ValueError("%s not a valid option for parameter "
                                 "importance!", p)
        for f in feature_analysis:
            if f not in ["box_violin", "correlation", "feat_importance",
                         "clustering", "feature_cdf"]:
                raise ValueError("%s not a valid option for feature analysis!", f)

        # Start analysis
        overview = self.analyzer.create_overview_table(self.best_run.folder)
        self.website["Meta Data"] = {
                     "table": overview,
                     "tooltip": "Meta data such as number of instances, "
                                "parameters, general configurations..." }

        compare_config = self.analyzer.config_to_html(self.default, self.incumbent)
        self.website["Best configuration"] = {"table": compare_config,
                     "tooltip": "Comparing parameters of default and incumbent. "
                                "Parameters that differ from default to "
                                "incumbent are presented first."}
        if  confviz and self.scenario.feature_array is not None:
            incumbents = [r.solver.incumbent for r in self.runs]
            confviz_script = self.analyzer.plot_confviz(incumbents)
            self.website["Configuration Visualization"] = {
                    "table" : confviz_script,
                    "tooltip" : "Using PCA to reduce dimensionality of the "
                                "search space  and plot the distribution of "
                                "evaluated configurations. The bigger the dot, "
                                "the more often the configuration was "
                                "evaluated. The colours refer to the predicted "
                                "performance in that part of the search space."}

        if performance:
            performance_table = self.analyzer.create_performance_table(
                                self.default, self.incumbent)
            self.website["Performance"] = {"table": performance_table}

        if cdf:
            cdf_path = self.analyzer.plot_cdf()
            self.website["Cumulative distribution function (CDF)"] = {
                     "figure": cdf_path,
                     "tooltip": "Plot default versus incumbent performance "
                                "on a cumulative distribution plot. Uses "
                                "validated data!"}

        if scatter and (self.scenario.train_insts != [[None]]):
            scatter_path = self.analyzer.plot_scatter()
            self.website["Scatterplot"] = {
                     "figure" : scatter_path,
                     "tooltip": "Plot all evaluated instances on a scatter plot, "
                                "to directly compare performance of incumbent "
                                "and default for each instance. Uses validated "
                                "data!"}
        elif scatter:
            self.logger.info("Scatter plot desired, but no instances available.")

        # Build report before time-consuming analysis
        builder.generate_html(self.website)

        elif confviz:
            self.logger.info("Configuration visualization desired, but no "
                             "instance-features available.")
        if cost_over_time:
            cost_over_time_path = self.analyzer.plot_cost_over_time(self.best_run.traj)
            self.website["Cost over time"] = {"figure": cost_over_time_path,
                    "tooltip": "The cost of the incumbent estimated over the "
                               "time. The cost is estimated using an EPM that "
                               "is based on the actual runs."}

        self.parameter_importance(ablation='ablation' in param_importance,
                                  fanova='fanova' in param_importance,
                                  forward_selection='forward_selection' in
                                                    param_importance)

        if parallel_coordinates:
            # Should be after parameter importance, if performed.
            self.logger.info("Plotting parallel coordinates.")
            n_params = 6
            parallel_path = self.analyzer.plot_parallel_coordinates(n_params)
            self.website["Parallel Coordinates"] = {
                         "figure" : parallel_path,
                         "tooltip": "Plot explored range of most important parameters."}

        self.feature_analysis(box_violin='box_violin' in feature_analysis,
                              correlation='correlation' in feature_analysis,
                              clustering='clustering' in feature_analysis)

        if algo_footprint:
            algo_footprint_path = self.analyzer.plot_algorithm_footprint()

        builder.generate_html(self.website)


    def parameter_importance(self, ablation=False, fanova=False,
                             forward_selection=False):
        """Perform the specified parameter importance procedures. """
        # PARAMETER IMPORTANCE
        if (ablation or forward_selection or fanova):
            self.website["Parameter Importance"] = OrderedDict([("tooltip",
                "Parameter Importance explains the individual importance of the "
                "parameters for the overall performance. Different techniques "
                "are implemented: fANOVA (functional analysis of "
                "variance), ablation and forward selection.")])
        if fanova:
            self.logger.info("fANOVA...")
            table, plots = self.analyzer.fanova(self.incumbent, 10)

            self.website["Parameter Importance"]["fANOVA"] = OrderedDict([
                ("tooltip", "fANOVA stands for functional analysis of variance "
                            "and predicts a parameters marginal performance, "
                            "by analyzing the predicted local neighbourhood of "
                            "this parameters optimized value, considering "
                            "correlations to other parameters and isolating "
                            "this parameters importance by predicting "
                            "performance changes that depend on other "
                            "parameters.")])

            self.website["Parameter Importance"]["fANOVA"]["Importance"] = {
                         "table": table}

            # Insert plots (the received plots is a dict, mapping param -> path)
            self.website["Parameter Importance"]["fANOVA"]["Marginals"] = OrderedDict([])
            for param, plot in plots.items():
                self.website["Parameter Importance"]["fANOVA"]["Marginals"][param] = {
                        "figure": plot}

        if ablation:
            self.logger.info("Ablation...")
            self.analyzer.parameter_importance("ablation", self.incumbent,
                                               self.output)
            ablationpercentage_path = os.path.join(self.output, "ablationpercentage.png")
            ablationperformance_path = os.path.join(self.output, "ablationperformance.png")
            self.website["Parameter Importance"]["Ablation (percentage)"] = {
                        "figure": ablationpercentage_path}
            self.website["Parameter Importance"]["Ablation (performance)"] = {
                        "figure": ablationperformance_path}
        if forward_selection:
            self.logger.info("Forward Selection...")
            self.analyzer.parameter_importance("forward-selection", self.incumbent,
                                               self.output)
            f_s_barplot_path = os.path.join(self.output, "forward-selection-barplot.png")
            f_s_chng_path = os.path.join(self.output, "forward-selection-chng.png")
            self.website["Parameter Importance"]["Forward Selection (barplot)"] = {
                        "figure": f_s_barplot_path}
            self.website["Parameter Importance"]["Forward Selection (chng)"] = {
                        "figure": f_s_chng_path}

    def feature_analysis(self, box_violin=False, correlation=False,
                         clustering=False):
        if not (box_violin or correlation or clustering):
            self.logger.debug("No feature analysis.")

        # FEATURE ANALYSIS (ASAPY)
        # TODO make the following line prettier
        # TODO save feature-names in smac
        in_reader = InputReader()
        feat_fn = self.scenario.feature_fn
        with changedir(self.ta_exec_dir):
            if not feat_fn or not os.path.exists(feat_fn):
                self.logger.warning("Feature Analysis needs valid feature "
                                    "file! Either {} is not a valid "
                                    "filename or features are not saved in "
                                    "the scenario.")
                self.logger.error("Skipping Feature Analysis.")
                return
            else:
                feat_names = in_reader.read_instance_features_file(self.scenario.feature_fn)[0]
        fa = FeatureAnalysis(output_dn=self.output,
                             scenario=self.scenario,
                             feat_names=feat_names)
        self.website["Feature Analysis"] = OrderedDict([])

        # box and violin plots
        if box_violin:
            name_plots = fa.get_box_violin_plots()
            self.website["Feature Analysis"]["Violin and box plots"] = OrderedDict({
                "tooltip": "Violin and Box plots to show the distribution of each instance feature. We removed NaN from the data."})
            for plot_tuple in name_plots:
                key = "%s" % (plot_tuple[0])
                self.website["Feature Analysis"]["Violin and box plots"][
                    key] = {"figure": plot_tuple[1]}


        # TODO: status_bar without scenario?
        #if "status_bar" in feature_analysis:
        #    status_plot = fa.get_bar_status_plot()
        #    self.website["Feature Analysis"]["Status Bar Plot"] = OrderedDict({
        #        "tooltip": "Stacked bar plots for runstatus of each feature groupe",
        #        "figure": status_plot})

        # correlation plot
        if correlation:
            correlation_plot = fa.correlation_plot()
            self.website["Feature Analysis"]["Correlation plot"] = {"tooltip": "Correlation based on Pearson product-moment correlation coefficients between all features and clustered with Wards hierarchical clustering approach. Darker fields corresponds to a larger correlation between the features.",
                                                            "figure": correlation_plot}
        # TODO
        #  File "/home/shuki/SpySMAC/spysmac/asapy/feature_analysis.py", line 197, in feature_importance
        #    pc.fit(scenario=self.scenario, config=config)
        #  File "/home/shuki/virtual-environments/spysmac/lib/python3.5/site-packages/autofolio/selector/pairwise_classification.py", line 66, in fit
        #    self.algorithms = scenario.algorithms
        #AttributeError: 'Scenario' object has no attribute 'algorithms'
        # feature importance
        #if "feat_importance" in feature_analysis:
        #    importance_plot = fa.feature_importance()
        #    self.website["Feature Analysis"]["Feature importance"] = {"tooltip": "Using the approach of SATZilla'11, we train a cost-sensitive random forest for each pair of algorithms and average the feature importance (using gini as splitting criterion) across all forests. We show the median, 25th and 75th percentiles across all random forests of the 15 most important features.",
        #                                                      "figure": importance_plot}

        # cluster instances in feature space
        if clustering:
            cluster_plot = fa.cluster_instances()
            self.website["Feature Analysis"]["Clustering"] = {"tooltip": "Clustering instances in 2d; the color encodes the cluster assigned to each cluster. Similar to ISAC, we use a k-means to cluster the instances in the feature space. As pre-processing, we use standard scaling and a PCA to 2 dimensions. To guess the number of clusters, we use the silhouette score on the range of 2 to 12 in the number of clusters",
                                                      "figure": cluster_plot}

        ## get cdf plot
        #if "feature_cdf" in feature_analysis:
        #    cdf_plot = fa.get_feature_cost_cdf_plot()
        #    self.website["Feature Analysis"]["CDF plot on feature costs"] = {"tooltip": "Cumulative Distribution function (CDF) plots. At each point x (e.g., running time cutoff), for how many of the instances (in percentage) have we computed the instance features. Faster feature computation steps have a higher curve. Missing values are imputed with the maximal value (or running time cutoff).",
        #                                                             "figure": cdf_plot}


