import os
import logging
from collections import OrderedDict
from contextlib import contextmanager
import typing
import copy

import numpy as np
from pandas import DataFrame

from smac.optimizer.objective import average_cost
from smac.runhistory.runhistory import RunHistory, DataOrigin
from smac.utils.io.input_reader import InputReader
from smac.utils.validate import Validator

from pimp.importance.importance import Importance

from cave.html.html_builder import HTMLBuilder
from cave.reader.configurator_run import ConfiguratorRun
from cave.analyzer import Analyzer
from cave.utils.helpers import scenario_sanity_check, combine_runhistories
from cave.utils.html_helpers import compare_configs_to_html
from cave.utils.timing import timing
from cave.utils.hpbandster2smac import HpBandSter2SMAC

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


class CAVE(object):
    def __init__(self,
                 folders: typing.List[str],
                 output_dir: str,
                 ta_exec_dir: typing.List[str],
                 file_format: str='SMAC3',
                 validation_format='NONE',
                 validation_method: str='epm',
                 pimp_max_samples: int=-1,
                 fanova_pairwise: bool=True,
                 use_budgets: bool=False,
                 seed: int=42):
        """
        Initialize CAVE facade to handle analyzing, plotting and building the report-page easily.
        During initialization, the analysis-infrastructure is built and the data is validated, the overall best
        incumbent is found and default+incumbent are evaluated for all instances for all runs, by default using an EPM.

        In the internal data-management the we have three types of runhistories: *original*, *validated* and *epm*.

        - *original* contain only runs that have been gathered during the optimization-process.
        - *validated* may contain original runs, but also data that was not gathered iteratively during the
          optimization, but systematically through external validation of interesting configurations.
          Important: NO ESTIMATED RUNS IN `validated` RUNHISTORIES!
        - *epm* contain runs that are gathered through empirical performance models.

        Runhistories are organized as follows:

        - each ConfiguratorRun has an *original_runhistory*- and a *combined_runhistory*-attribute
        - if available, each ConfiguratorRun's *validated_runhistory* contains
          a runhistory with validation-data gathered after the optimization
        - *combined_runhistory* always contains as many real runs as possible

        CaveFacade contains three runhistories:

        - *original_rh*: original runs that have been performed **during optimization**!
        - *validated_rh*: runs that have been validated, so they were not part of the original optimization
        - *epm_rh*: contains epm-predictions for all incumbents

        The analyze()-method performs an analysis and output a report.html.

        Arguments
        ---------
        folders: list<strings>
            paths to relevant SMAC runs
        output_dir: string
            output for cave to write results (figures + report)
        ta_exec_dir: string
            execution directory for target algorithm (to find instance.txt specified in scenario, ..)
        file_format: str
            what format the rundata is in, options are [SMAC3, SMAC2 and CSV]
        validation_method: string
            from [validation, epm], how to estimate missing runs
        pimp_max_samples: int
            passed to PIMP for configuration
        fanova_pairwise: bool
            whether to calculate pairwise marginals for fanova
        use_budgets: bool
            if true, individual runs are treated as different budgets. they are not evaluated together, but compared
            against each other. runs are expected in ascending budget-size.
        seed: int
            random seed for analysis (e.g. the random forests)
        """
        self.logger = logging.getLogger(self.__module__ + '.' + self.__class__.__name__)
        self.output_dir = output_dir
        self.rng = np.random.RandomState(seed)
        self.use_budgets = use_budgets
        self.ta_exec_dir = ta_exec_dir
        self.file_format = file_format
        self.validation_format = validation_format
        self.validation_method = validation_method
        self.pimp_max_samples = pimp_max_samples
        self.fanova_pairwise = fanova_pairwise

        self.bohb_result = None  # only relevant for bohb_result

        # Create output_dir if necessary
        self.logger.info("Saving results to '%s'", self.output_dir)
        if not os.path.exists(output_dir):
            self.logger.debug("Output-dir '%s' does not exist, creating", self.output_dir)
            os.makedirs(output_dir)

        if file_format == 'BOHB':
            if len(folders) != 1:
                raise ValueError("For file format BOHB you can only specify one folder.")
            self.bohb_result, folders = HpBandSter2SMAC().convert(folders[0])

        # Save all relevant configurator-runs in a list
        self.logger.debug("Folders: %s; ta-exec-dirs: %s", str(folders), str(ta_exec_dir))
        self.runs = []
        if len(ta_exec_dir) < len(folders):
            for i in range(len(folders) - len(ta_exec_dir)):
                ta_exec_dir.append(ta_exec_dir[0])
        for ta_exec_dir, folder in zip(ta_exec_dir, folders):
            try:
                self.logger.debug("Collecting data from %s.", folder)
                self.runs.append(ConfiguratorRun(folder, ta_exec_dir, file_format=file_format,
                                                 validation_format=validation_format))
            except Exception as err:
                self.logger.warning("Folder %s could with ta_exec_dir %s not be loaded, failed with error message: %s",
                                    folder, ta_exec_dir, err)
                self.logger.exception(err)
                continue
        if not self.runs:
            raise ValueError("None of the specified folders could be loaded.")

        # Use scenario of first run for general purposes (expecting they are all the same anyway!
        self.scenario = self.runs[0].solver.scenario
        scenario_sanity_check(self.scenario, self.logger)
        self.default = self.scenario.cs.get_default_configuration()

        # All runs that have been actually explored during optimization
        self.global_original_rh = None
        # All original runs + validated runs if available
        self.global_validated_rh = None
        # All validated runs + EPM-estimated for def and inc on all insts
        self.global_epm_rh = None
        self.pimp = None
        self.model = None

        if use_budgets:
            self._init_helper_budgets()
        else:
            self._init_helper_no_budgets()

        self.analyzer = Analyzer(self.default, self.incumbent,
                                 self.scenario, self.output_dir,
                                 pimp_max_samples, fanova_pairwise,
                                 rng=self.rng)


        # Builder for html-website
        custom_logo = './custom_logo.png'
        if file_format.startswith('SMAC'):
            logo_fn = 'SMAC_logo.png'
        elif file_format == 'BOHB':
            logo_fn = 'BOHB_logo.png'
        elif os.path.exists(custom_logo):
            logo_fn = custom_logo
        else:
            logo_fn = 'ml4aad.png'
            self.logger.info("No suitable logo found. You can use a custom logo simply by having a file called '%s' "
                             "in the directory from which you run CAVE.", custom_logo)
        self.builder = HTMLBuilder(self.output_dir, "CAVE", logo_fn=logo_fn, logo_custom=custom_logo==logo_fn)
        self.website = OrderedDict([])

    def _init_helper_budgets(self):
        self.best_run = self.runs[-1]
        self.incumbent = self.best_run.solver.incumbent

    def _init_helper_no_budgets(self):
        """No budgets means using global, aggregated runhistories to analyze the Configurator's behaviour.
        Also it creates an EPM using all available information, since all runs are "equal".
        """
        self.global_original_rh = RunHistory(average_cost)
        self.global_validated_rh = RunHistory(average_cost)
        self.global_epm_rh = RunHistory(average_cost)# Save all relevant SMAC-runs in a list
        self.logger.debug("Update original rh with all available rhs!")
        for run in self.runs:
            self.global_original_rh.update(run.original_runhistory, origin=DataOrigin.INTERNAL)
            self.global_validated_rh.update(run.original_runhistory, origin=DataOrigin.INTERNAL)
            if run.validated_runhistory:
                self.global_validated_rh.update(run.validated_runhistory, origin=DataOrigin.EXTERNAL_SAME_INSTANCES)

        self._init_pimp_and_validator(self.global_validated_rh)

        # Estimate missing costs for [def, inc1, inc2, ...]
        self.validate_default_and_incumbents(self.validation_method, self.ta_exec_dir)
        self.global_epm_rh.update(self.global_validated_rh)

        for rh_name, rh in [("original", self.global_original_rh),
                            ("validated", self.global_validated_rh),
                            ("epm", self.global_epm_rh)]:
            self.logger.debug('Combined number of RunHistory data points for %s runhistory: %d '
                              '# Configurations: %d. # Configurator runs: %d',
                              rh_name, len(rh.data), len(rh.get_all_configs()), len(self.runs))

        # Sort runs (best first)
        self.runs = sorted(self.runs, key=lambda run: self.global_epm_rh.get_cost(run.solver.incumbent))
        self.best_run = self.runs[0]

        self.incumbent = self.pimp.incumbent = self.best_run.solver.incumbent
        self.logger.debug("Overall best run: %s, with incumbent: %s", self.best_run.folder, self.incumbent)

    def _init_pimp_and_validator(self, rh, alternative_output_dir=None):
        """Create ParameterImportance-object and use it's trained model for  validation and further predictions
        We pass validated runhistory, so that the returned model will be based on as much information as possible

        Parameters
        ----------
        rh: RunHistory
            runhistory used to build EPM
        alternative_output_dir: str
            e.g. for budgets we want pimp to use an alternative output-dir (subfolders per budget)
        """
        self.logger.debug("Using '%s' as output for pimp", alternative_output_dir if alternative_output_dir else
                self.output_dir)
        self.pimp = Importance(scenario=copy.deepcopy(self.scenario),
                               runhistory=rh,
                               incumbent=self.default,  # Inject correct incumbent later
                               parameters_to_evaluate=4,
                               save_folder=alternative_output_dir if alternative_output_dir else self.output_dir,
                               seed=self.rng.randint(1, 100000),
                               max_sample_size=self.pimp_max_samples,
                               fANOVA_pairwise=self.fanova_pairwise,
                               preprocess=False)
        self.model = self.pimp.model

        # Validator (initialize without trajectory)
        self.validator = Validator(self.scenario, None, None)
        self.validator.epm = self.model

    @timing
    def validate_default_and_incumbents(self, method, ta_exec_dir):
        """Validate default and incumbent configurations on all instances possible.
        Either use validation (physically execute the target algorithm) or EPM-estimate and update according runhistory
        (validation -> self.global_validated_rh; epm -> self.global_epm_rh).

        Parameters
        ----------
        method: str
            epm or validation
        ta_exec_dir: str
            path from where the target algorithm can be executed as found in scenario (only used for actual validation)
        """
        for run in self.runs:
            self.logger.debug("Validating %s using %s!", run.folder, method)
            self.validator.traj = run.traj
            if method == "validation":
                with changedir(ta_exec_dir):
                    # TODO determine # repetitions
                    new_rh = self.validator.validate('def+inc', 'train+test', 1, -1, runhistory=self.global_validated_rh)
                self.global_validated_rh.update(new_rh)
            elif method == "epm":
                # Only do test-instances if features for test-instances are available
                instance_mode = 'train+test'
                if (any([i not in self.scenario.feature_dict for i in self.scenario.test_insts]) and
                    any([i in self.scenario.feature_dict for i in self.scenario.train_insts])):  # noqa
                    self.logger.debug("No features provided for test-instances (but for train!). "
                                      "Cannot validate on \"epm\".")
                    self.logger.warning("Features detected for train-instances, but not for test-instances. This is "
                                        "unintended usage and may lead to errors for some analysis-methods.")
                    instance_mode = 'train'

                new_rh = self.validator.validate_epm('def+inc', instance_mode, 1, runhistory=self.global_validated_rh)
                self.global_epm_rh.update(new_rh)
            else:
                raise ValueError("Missing data method illegal (%s)", method)
            self.validator.traj = None  # Avoid usage-mistakes

    @timing
    def analyze(self,
                performance=True,
                cdf=True,
                scatter=True,
                cfp=True,
                cfp_time_slider=False,
                cfp_max_plot=-1,
                cfp_number_quantiles=10,
                param_importance=['forward_selection', 'ablation', 'fanova'],
                pimp_sort_table_by: str="average",
                feature_analysis=["box_violin", "correlation", "importance", "clustering", "feature_cdf"],
                parallel_coordinates=True,
                cost_over_time=True,
                algo_footprint=True):
        """Analyze the available data and build HTML-webpage as dict.
        Save webpage in 'self.output_dir/CAVE/report.html'.
        Analyzing is performed with the analyzer-instance that is initialized in
        the __init__

        Parameters
        ----------
        performance: bool
            whether to calculate par10-values
        cdf: bool
            whether to plot cdf
        scatter: bool
            whether to plot scatter
        cfp: bool
            whether to perform configuration visualization
        cfp_time_slider: bool
            whether to include an interactive time-slider in configuration footprint
        cfp_max_plot: int
            limit number of configurations considered for configuration footprint (-1 -> all configs)
        cfp_number_quantiles: int
            number of steps over time generated in configuration footprint
        param_importance: List[str]
            containing methods for parameter importance
        pimp_sort_table: str
            in what order the parameter-importance overview should be organized
        feature_analysis: List[str]
            containing methods for feature analysis
        parallel_coordinates: bool
            whether to plot parallel coordinates
        cost_over_time: bool
            whether to plot cost over time
        algo_footprint: bool
            whether to plot algorithm footprints
        """
        # Check arguments
        for p in param_importance:
            if p not in ['forward_selection', 'ablation', 'fanova', 'lpi']:
                raise ValueError("%s not a valid option for parameter importance!" % p)
        for f in feature_analysis:
            if f not in ["box_violin", "correlation", "importance", "clustering", "feature_cdf"]:
                raise ValueError("%s not a valid option for feature analysis!" % f)

        # Start analysis


        headings = ["Meta Data",
                    "Best Configuration",
                    "Performance Analysis",
                    "Configurator's Behavior",
                    "Parameter Importance",
                    "Feature Analysis"
                    ]
        for h in headings:
            self.website[h] = OrderedDict()

        if self.use_budgets:
            # The individual configurator runs are not directory comparable and cannot be aggregated.
            # Nevertheless they need to be combined in one comprehensive report and some metrics are to be compared over
            # the individual runs.

            # if self.file_format == 'BOHB':
            #     self.website["BOHB Visualization"] = {"figure" : [self.analyzer.bohb_plot(self.bohb_result)]}

            # Perform analysis for each run
            for run in self.runs:
                sub_sec = os.path.basename(run.folder)
                # Set paths for each budget individual to avoid path-conflicts
                sub_output_dir = os.path.join(self.output_dir, 'content', sub_sec)
                os.makedirs(sub_output_dir, exist_ok=True)
                self.analyzer = Analyzer(run.default, run.incumbent,
                                         self.scenario, sub_output_dir,
                                         self.pimp_max_samples, self.fanova_pairwise,
                                         rng=self.rng)
                # Set runhistories
                self.global_original_rh = run.original_runhistory
                self.global_validated_rh = run.combined_runhistory
                self.global_epm_rh = RunHistory(average_cost)
                # Train epm and stuff
                self._init_pimp_and_validator(run.combined_runhistory, alternative_output_dir=sub_output_dir)
                self.validate_default_and_incumbents(self.validation_method, run.ta_exec_dir)
                self.pimp.incumbent = run.incumbent
                self.incumbent = run.incumbent
                run.epm_rh = self.global_epm_rh
                self.best_run = run
                # Perform analysis
                overview = self.analyzer.create_overview_table(self.global_original_rh,
                                                               run, len(self.runs), self.default, self.incumbent)
                self.website["Meta Data"][sub_sec] = {"table": overview}
                compare_config_html = compare_configs_to_html(self.default, self.incumbent)
                self.website["Best Configuration"][sub_sec] = {"table": compare_config_html}

                d = self.website["Performance Analysis"][sub_sec] = OrderedDict()
                self.performance_analysis(d, performance, cdf, scatter, algo_footprint)
                d = self.website["Parameter Importance"][sub_sec] = OrderedDict()
                self.parameter_importance(d, ablation='ablation' in param_importance,
                                          fanova='fanova' in param_importance,
                                          forward_selection='forward_selection' in param_importance,
                                          lpi='lpi' in param_importance,
                                          pimp_sort_table_by=pimp_sort_table_by)
                d = self.website["Configurator's Behavior"][sub_sec] = OrderedDict()
                self.configurators_behavior(d, cost_over_time,
                                            cfp, cfp_max_plot, cfp_time_slider, cfp_number_quantiles,
                                            parallel_coordinates)
                d = self.website["Feature Analysis"][sub_sec] = OrderedDict()
                self.feature_analysis(d, box_violin='box_violin' in feature_analysis,
                                      correlation='correlation' in feature_analysis,
                                      clustering='clustering' in feature_analysis,
                                      importance='importance' in feature_analysis)
                self.original_runhistory = self.validated_runhistory = self.epm_runhistory = None
        else:
            overview = self.analyzer.create_overview_table(self.global_original_rh,
                                                           self.runs[0], len(self.runs), self.default, self.incumbent)
            self.website["Meta Data"] = {"table": overview}
            compare_config_html = compare_configs_to_html(self.default, self.incumbent)
            self.website["Best Configuration"] = {"table": compare_config_html}

            self.performance_analysis(self.website["Performance Analysis"], performance, cdf, scatter, algo_footprint)
            self.parameter_importance(self.website["Parameter Importance"],
                                      ablation='ablation' in param_importance,
                                      fanova='fanova' in param_importance,
                                      forward_selection='forward_selection' in param_importance,
                                      lpi='lpi' in param_importance,
                                      pimp_sort_table_by=pimp_sort_table_by)
            self.configurators_behavior(self.website["Configurator's Behavior"], cost_over_time,
                                        cfp, cfp_max_plot, cfp_time_slider, cfp_number_quantiles,
                                        parallel_coordinates)
            self.feature_analysis(self.website["Feature Analysis"],
                                  box_violin='box_violin' in feature_analysis,
                                  correlation='correlation' in feature_analysis,
                                  clustering='clustering' in feature_analysis,
                                  importance='importance' in feature_analysis)

        self.build_website()

        self.logger.info("CAVE finished. Report is located in %s",
                         os.path.join(self.output_dir, 'report.html'))

    def performance_analysis(self, d,
                             performance, cdf, scatter, algo_footprint):
        """Generate performance analysis.

        Parameters
        ----------
        d: dictionary
            dictionary to add entries to

        performance, cdf, scatter, algo_footprint: bool
            what analysis-methods to perform
        """

        if performance:
            instances = [i for i in self.scenario.train_insts + self.scenario.test_insts if i]
            oracle = self.analyzer.get_oracle(instances, self.global_validated_rh)
            performance_table = self.analyzer.create_performance_table(self.default, self.incumbent,
                                                                       self.global_epm_rh, oracle)
            d["Performance Table"] = {"table": performance_table}

        if cdf:
            cdf_paths = self.analyzer.plot_cdf_compare(self.default, self.incumbent, self.global_epm_rh)
            if cdf_paths:
                d["empirical Cumulative Distribution Function (eCDF)"] = {"figure": cdf_paths}

        if scatter:
            scatter_paths = self.analyzer.plot_scatter(self.default, self.incumbent, self.global_epm_rh)
            if scatter_paths:
                d["Scatterplot"] = {"figure": scatter_paths}
            self.build_website()

        if algo_footprint and self.scenario.feature_dict:
            algorithms = [(self.default, "default"), (self.incumbent, "incumbent")]

            algo_footprint_plots = self.analyzer.plot_algorithm_footprint(self.global_epm_rh, algorithms)
            d["Algorithm Footprints"] = OrderedDict()

            # Interactive bokeh-plot
            script, div = algo_footprint_plots[0]
            d["Algorithm Footprints"]["Interactive Algorithm Footprint"] = {"bokeh" : (script, div)}

            p_3d = algo_footprint_plots[1]
            for plots in p_3d:
                header = os.path.splitext(os.path.split(plots[0])[1])[0][10:-2]
                header = header[0].upper() + header[1:].replace('_', ' ')
                d["Algorithm Footprints"][header] = {"figure_x2": plots}

        self.build_website()

    def configurators_behavior(self,
                               d,
                               cost_over_time=False,
                               cfp=False,
                               cfp_max_plot=-1,
                               cfp_time_slider=False,
                               cfp_number_quantiles=1,
                               parallel_coordinates=False):

        if cost_over_time:
            cost_over_time_script = self.analyzer.plot_cost_over_time(self.global_validated_rh, self.runs, self.validator)
            d["Cost Over Time"] = {"bokeh": cost_over_time_script}
            self.build_website()

        if cfp:  # Configurator Footprint
            runs = [self.best_run] if self.use_budgets else self.runs
            res = self.analyzer.plot_configurator_footprint(self.scenario, runs, self.global_original_rh,
                                                            max_confs=cfp_max_plot,
                                                            time_slider=(cfp_time_slider and
                                                                         (cfp_number_quantiles > 1)),
                                                            num_quantiles=cfp_number_quantiles)
            bokeh_components, cfp_paths = res
            if cfp_number_quantiles == 1:  # Only one plot, no need for "Static"-field
                d["Configurator Footprint"] = {"bokeh": (bokeh_components)}
            else:
                d["Configurator Footprint"] = OrderedDict()
                d["Configurator Footprint"]["Interactive"] = {"bokeh": (bokeh_components)}
                if all([True for p in cfp_paths if os.path.exists(p)]):  # If the plots were actually generated
                    d["Configurator Footprint"]["Static"] = {"figure": cfp_paths}
                else:
                    d["Configurator Footprint"]["Static"] = {
                            "else": "This plot is missing. Maybe it was not generated? "
                                    "Check if you installed selenium and phantomjs "
                                    "correctly to activate bokeh-exports. "
                                    "(https://automl.github.io/CAVE/stable/faq.html)"}

            self.build_website()

        if parallel_coordinates:
            # Should be after parameter importance, if performed.
            n_params = 6
            parallel_path = self.analyzer.plot_parallel_coordinates(self.global_original_rh, self.global_validated_rh,
                                                                    self.validator, n_params)
            if parallel_path:
                d["Parallel Coordinates"] = {"figure": parallel_path}
            self.build_website()

    def parameter_importance(self, d, ablation=False, fanova=False,
                             forward_selection=False, lpi=False, pimp_sort_table_by='average'):
        """Perform the specified parameter importance procedures. """

        sum_ = 0
        if fanova:
            sum_ += 1
            self.logger.info("fANOVA...")
            d["fANOVA"] = OrderedDict()

            try:
                table, plots, pair_plots = self.analyzer.fanova(self.pimp, self.incumbent)

                d["fANOVA"]["Importance"] = {"table": table}

                # Insert plots (the received plots is a dict, mapping param -> path)
                d["fANOVA"]["Marginals"] = OrderedDict()
                for param, plot in plots.items():
                    d["fANOVA"]["Marginals"][param] = {"figure": plot}
                if pair_plots:
                    d["fANOVA"]["Pairwise Marginals"] = OrderedDict()
                    for param, plot in pair_plots.items():
                        d["fANOVA"]["Pairwise Marginals"][param] = {"figure": plot}
            except RuntimeError as e:
                err = "Encountered error '%s' in fANOVA, this can e.g. happen with too few data-points." % e
                self.logger.exception(err)
                d["fANOVA"] = {"else": err + " Check 'debug/debug.log' for more information."}

            self.build_website()

        if ablation:
            sum_ += 1
            self.logger.info("Ablation...")
            self.analyzer.parameter_importance(self.pimp, "ablation", self.incumbent, self.analyzer.output_dir)
            ablationpercentage_path = os.path.join(self.analyzer.output_dir, "ablationpercentage.png")
            ablationperformance_path = os.path.join(self.analyzer.output_dir, "ablationperformance.png")
            d["Ablation"] = {"figure": [ablationpercentage_path, ablationperformance_path]}
            self.build_website()

        if forward_selection:
            sum_ += 1
            self.logger.info("Forward Selection...")
            self.analyzer.parameter_importance(self.pimp, "forward-selection", self.incumbent, self.analyzer.output_dir)
            f_s_barplot_path = os.path.join(self.analyzer.output_dir, "forward-selection-barplot.png")
            f_s_chng_path = os.path.join(self.analyzer.output_dir, "forward-selection-chng.png")
            d["Forward Selection"] = {"figure": [f_s_barplot_path, f_s_chng_path]}
            self.build_website()

        if lpi:
            sum_ += 1
            self.logger.info("Local EPM-predictions around incumbent...")
            plots = self.analyzer.local_epm_plots(self.pimp)
            d["Local Parameter Importance (LPI)"] = OrderedDict()
            for param, plot in plots.items():
                d["Local Parameter Importance (LPI)"][param] = {"figure": plot}
            self.build_website()

        if sum_ >= 2:
            out_fn = os.path.join(self.output_dir, 'pimp.tex')
            self.logger.info('Creating pimp latex table at %s' % out_fn)
            self.pimp.table_for_comparison(self.analyzer.evaluators, out_fn, style='latex')
            table = self.analyzer.importance_table(pimp_sort_table_by)
            d["Importance Table"] = {
                    "table": table,
                    "tooltip": "Parameters are sorted by {}. Note, that the values are not "
                               "directly comparable, since the different techniques "
                               "provide different metrics (see respective tooltips "
                               "for details on the differences).".format(pimp_sort_table_by)}
            d.move_to_end("Importance Table", last=False)
            self.build_website()

    def feature_analysis(self, d, box_violin=False, correlation=False, clustering=False, importance=False):

        if not self.scenario.feature_dict:
            self.logger.error("No features available. Skipping feature analysis.")
            return
        feat_fn = self.scenario.feature_fn
        if not self.scenario.feature_names:
            self.logger.debug("`scenario.feature_names` is not set. Loading from '%s'", feat_fn)
            with changedir(self.ta_exec_dir if self.ta_exec_dir else '.'):
                if not feat_fn or not os.path.exists(feat_fn):
                    self.logger.warning("Feature names are missing. Either provide valid feature_file in scenario "
                                        "(currently %s) or set `scenario.feature_names` manually." % feat_fn)
                    self.logger.error("Skipping Feature Analysis.")
                    return
                else:
                    # Feature names are contained in feature-file and retrieved
                    feat_names = InputReader().read_instance_features_file(feat_fn)[0]
        else:
            feat_names = copy.deepcopy(self.scenario.feature_names)

        # feature importance using forward selection
        if importance:
            d["Feature Importance"] = OrderedDict()
            imp, plots = self.analyzer.feature_importance(self.pimp)
            imp = DataFrame(data=list(imp.values()), index=list(imp.keys()), columns=["Error"])
            imp = imp.to_html()  # this is a table with the values in html
            d["Feature Importance"]["Table"] = {"table": imp}
            for p in plots:
                name = os.path.splitext(os.path.basename(p))[0]
                d["Feature Importance"][name] = {"figure": p}

        # box and violin plots
        if box_violin:
            name_plots = self.analyzer.feature_analysis('box_violin', feat_names)
            d["Violin and Box Plots"] = OrderedDict()
            for plot_tuple in name_plots:
                key = "%s" % (plot_tuple[0])
                d["Violin and Box Plots"][key] = {"figure": plot_tuple[1]}

        # correlation plot
        if correlation:
            correlation_plot = self.analyzer.feature_analysis('correlation', feat_names)
            if correlation_plot:
                d["Correlation"] = {"figure": correlation_plot}

        # cluster instances in feature space
        if clustering:
            cluster_plot = self.analyzer.feature_analysis('clustering', feat_names)
            d["Clustering"] = {"figure": cluster_plot}

        self.build_website()

    def build_website(self):
        self.builder.generate_html(self.website)
