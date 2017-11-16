import os
import logging
from collections import OrderedDict
from contextlib import contextmanager
import typing
import json
import operator
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
        is built and the data is validated, that means the overall best
        incumbent is found and default+incumbent are evaluated for all
        instances, usually using an EPM.
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
        # only estimated using an EPM for further EPMs.
        self.global_rh = RunHistory(average_cost)

        # Save all relevant SMAC-runs in a list
        self.runs = []
        for folder in folders:
            if not os.path.exists(folder):
                raise ValueError("The specified SMAC-output in %s doesn't exist.",
                                 folder)
            self.logger.debug("Collecting data from %s.", folder)
            self.runs.append(SMACrun(folder, ta_exec_dir))

        # Use scenario of first run for general purposes (expecting they are all
        # the same anyway!)
        self.scenario = self.runs[0].solver.scenario

        # Update global runhistory with all available runhistories
        self.logger.debug("Update global rh with all available rhs!")
        runhistory_fns = [os.path.join(f, "runhistory.json") for f in folders]
        for rh_file in runhistory_fns:
            self.global_rh.update_from_json(rh_file, self.scenario.cs)
        self.logger.debug('Combined number of Runhistory data points: %d. '
                          '# Configurations: %d. # Runhistories: %d',
                          len(runhistory_fns), len(self.global_rh.data),
                          len(self.global_rh.get_all_configs()))
        # Keep copy for plots/analysis that require only "pure" data (such as
        # visualizing the actual explored configspace)
        self.global_rh_actual = copy.deepcopy(self.global_rh)

        # Estimate all missing costs using validation or EPM
        self.complete_data(method=missing_data_method)
        self.best_run = min(self.runs, key=lambda run:
                self.global_rh.get_cost(run.solver.incumbent))

        # TODO for ablation and forward selection inject in self.imp
        self.importance = None  # Used to store dictionary containing parameter
                                # importances, so it can be used by analysis

        self.default = self.scenario.cs.get_default_configuration()
        self.incumbent = self.best_run.solver.incumbent

        # Following variable determines whether a distinction is made
        # between train and testinstance (e.g. in plotting)
        self.train_test = bool(self.scenario.train_insts != [None] and
                               self.scenario.test_insts != [None])

        self.analyzer = Analyzer(self.global_rh, self.train_test, self.scenario)
        conf1_runs = get_cost_dict_for_config(self.global_rh, self.default)
        conf2_runs = get_cost_dict_for_config(self.global_rh, self.incumbent)
        self.plotter = Plotter(self.scenario, self.train_test, conf1_runs, conf2_runs)
        self.website = OrderedDict([])

    def complete_data(self, method="epm", c_runs=None):
        """Complete missing data of runs to be analyzed. Either using validation
        or EPM.
        """
        if not c_runs:
            c_runs = self.runs
        with changedir(self.ta_exec_dir):
            self.logger.info("Completing data using %s.", method)

            path_for_validated_rhs = os.path.join(self.output, "validated_rhs")
            for run in c_runs:
            #for run in [self.best_run]:
                out = os.path.join(path_for_validated_rhs, "rh_"+run.folder)
                validator = Validator(run.scen, run.traj, out)

                if method == "validation":
                    # TODO determine # repetitions
                    self.global_rh.update(validator.validate('def+inc',
                                          'train+test', 1, -1,
                                          runhistory=self.global_rh))
                elif method == "epm":
                    self.global_rh.update(validator.validate_epm('def+inc', 'train+test', 1,
                                             runhistory=self.global_rh))
                else:
                    raise ValueError("Missing data method illegal (%s)",
                                     method)

    def analyze(self,
                performance=False, cdf=False, scatter=False, confviz=False,
                param_importance=['forward_selection', 'ablation', 'fanova'],
                feature_analysis=["box_violin", "correlation",
                    "feat_importance", "clustering", "feature_cdf"],
                parallel_coordinates=True):
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

        best_config = self.analyzer.config_to_html(self.default, self.incumbent)
        self.website["Best configuration"] = {"table": best_config,
                     "tooltip": "Comparing parameters of default and incumbent. "
                                "Parameters that differ from default to "
                                "incumbent are presented first."}

        if performance:
            performance_table = self.analyzer.create_performance_table(
                                self.default, self.incumbent)
            self.website["Performance"] = {"table": performance_table}

        if cdf:
            cdf_path = os.path.join(self.output, 'cdf.png')
            self.plotter.plot_cdf_compare(output=cdf_path)
            self.website["Cumulative distribution function (CDF)"] = {
                     "figure": cdf_path,
                     "tooltip": "Plot default versus incumbent performance "
                                "on a cumulative distribution plot."}

        if scatter and (self.scenario.train_insts != [[None]]):
            scatter_path = os.path.join(self.output, 'scatter.png')
            self.plotter.plot_scatter(output=scatter_path)
            self.website["Scatterplot"] = {
                     "figure" : scatter_path,
                     "tooltip": "Plot all evaluated instances on a scatter plot, "
                                "to directly compare performance of incumbent "
                                "and default for each instance."}
        elif scatter:
            self.logger.info("Scatter plot desired, but no instances available.")

        if  confviz and self.scenario.feature_array is not None:
            incumbents = [r.solver.incumbent for r in self.runs]
            confviz = self.plotter.visualize_configs(self.scenario,
                        self.global_rh, incumbents)
            self.website["Configuration Visualization"] = {
                    "table" : confviz,
                    "tooltip" : "Using PCA to reduce dimensionality of the "
                                "search space  and plot the distribution of "
                                "evaluated configurations. The bigger the dot, "
                                "the more often the configuration was "
                                "evaluated. The colours refer to the predicted "
                                "performance in that part of the search space."}
        elif confviz:
            self.logger.info("Configuration visualization desired, but no "
                             "instance-features available.")

        self.parameter_importance(ablation='ablation' in param_importance,
                                  fanova='fanova' in param_importance,
                                  forward_selection='forward_selection' in
                                                    param_importance)

        if parallel_coordinates:
            self.logger.info("Plotting parallel coordinates.")
            out_path = os.path.join(self.output, "parallel_coordinates.png")
            # TODO what if no parameter importance is done? plot all? random subset?
            if self.importance:
                params = list(self.importance.keys())[:6]
            else:
                params = list(self.default.keys())
            self.logger.debug("Parallel coordinates plotting params: " + str(params))
            self.plotter.plot_parallel_coordinates(self.global_rh, out_path,
                    params)
            self.website["Parallel Coordinates"] = {
                     "figure" : out_path,
                     "tooltip": "Plot explored range of most important parameters."}

        self.plotter.plot_cost_over_time(self.global_rh, self.best_run.traj)

        self.feature_analysis(box_violin='box_violin' in feature_analysis,
                              correlation='correlation' in feature_analysis,
                              clustering='clustering' in feature_analysis)

        footprint = AlgorithmFootprint(self.global_rh, self.scenario.feature_dict,
                                       self.scenario.cutoff)
        footprint.plot_points(self.incumbent, "inc.png")
        footprint.plot_points(self.default, "def.png")


    def parameter_importance(self, ablation=False, fanova=False,
                             forward_selection=False):
        """Perform the specified parameter importance procedures. """
        # PARAMETER IMPORTANCE
        if (ablation or forward_selection or fanova):
            self.website["Parameter Importance"] = OrderedDict([("tooltip",
                "Parameter Importance explains the individual importance of the "
                "parameters for the overall performance. Different techniques "
                "are implemented, for example: fANOVA (functional analysis of "
                "variance), ablation and forward selection.")])
        if fanova:
            self.logger.info("fANOVA...")
            params = self.analyzer.fanova(self.incumbent, self.output, 10)
            self.importance = params

            self.website["Parameter Importance"]["fANOVA"] = OrderedDict([
                ("tooltip", "fANOVA stands for functional analysis of variance "
                            "and predicts a parameters marginal performance, "
                            "by analyzing the predicted local neighbourhood of "
                            "this parameters optimized value, considering "
                            "correlations to other parameters and isolating "
                            "this parameters importance by predicting "
                            "performance changes that depend on other "
                            "parameters.")])

            # Create table
            fanova_table = self.analyzer._split_table(params)
            df = DataFrame(data=fanova_table)
            fanova_table = df.to_html(escape=False, header=False, index=False, justify='left')
            self.website["Parameter Importance"]["fANOVA"]["Importance"] = {
                        "table": fanova_table}

            # Insert plots
            self.website["Parameter Importance"]["fANOVA"]["Marginals"] = OrderedDict([])
            for p in [x[0] for x in sorted(params.items(),
                key=operator.itemgetter(1))]:
                self.website["Parameter Importance"]["fANOVA"]["Marginals"][p] = {
                        "figure": os.path.join(self.output, "fanova", p+'.png')}
            # Check for pairwise plots (untested and hacky TODO)
            # Right now no way to access paths of the plots -> file issue
            pairwise = OrderedDict([])
            for p1 in params.keys():
                for p2 in params.keys():
                    combi = str([p1, p2]).replace(os.sep, "_").replace("'","") + ".png"
                    potential_path = os.path.join(self.output, 'fanova', combi)
                    if os.path.exists(potential_path):
                         pairwise[combi] = {"figure": potential_path}
            if pairwise:
                self.website["Parameter Importance"]["fANOVA"]["PairwiseMarginals"] = pairwise

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
        ## feature importance
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

        builder = HTMLBuilder(self.output, "SpySMAC")
        builder.generate_html(self.website)

