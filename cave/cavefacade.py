import os
import logging
from collections import OrderedDict
from contextlib import contextmanager
import typing
from typing import Union, List
import copy
from functools import wraps
import shutil

import numpy as np
from pandas import DataFrame

from smac.optimizer.objective import average_cost
from smac.runhistory.runhistory import RunHistory, DataOrigin
from smac.utils.io.input_reader import InputReader
from smac.utils.validate import Validator

from pimp.importance.importance import Importance

from cave.html.html_builder import HTMLBuilder
from cave.reader.configurator_run import ConfiguratorRun
from cave.utils.helpers import scenario_sanity_check, combine_runhistories
from cave.utils.timing import timing
from cave.utils.hpbandster2smac import HpBandSter2SMAC
from cave.analyzer.cost_over_time import CostOverTime
from cave.analyzer.parallel_coordinates import ParallelCoordinates
from cave.analyzer.configurator_footprint import ConfiguratorFootprint
from cave.analyzer.performance_table import PerformanceTable
from cave.analyzer.plot_ecdf import PlotECDF
from cave.analyzer.plot_scatter import PlotScatter
from cave.analyzer.algorithm_footprint import AlgorithmFootprint
from cave.analyzer.cave_fanova import CaveFanova
from cave.analyzer.cave_ablation import CaveAblation
from cave.analyzer.cave_forward_selection import CaveForwardSelection
from cave.analyzer.local_parameter_importance import LocalParameterImportance
from cave.analyzer.pimp_comparison_table import PimpComparisonTable
from cave.analyzer.feature_importance import FeatureImportance
from cave.analyzer.box_violin import BoxViolin
from cave.analyzer.feature_correlation import FeatureCorrelation
from cave.analyzer.feature_clustering import FeatureClustering
from cave.analyzer.overview_table import OverviewTable
from cave.analyzer.compare_default_incumbent import CompareDefaultIncumbent
from cave.analyzer.bohb_learning_curves import BohbLearningCurves
from cave.__version__ import __version__ as v

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

def analyzer_type(f):
    @wraps(f)
    def wrap(self, *args, d=None, **kw):
        run = kw.pop('run', None)
        if self.use_budgets and not f.__name__ in self.always_aggregated:
            if run:
                # Use the run-specific cave instance
                try:
                    cave = self.folder_to_run[run].cave
                except KeyError as err:
                    raise KeyError("You specified '%s' as folder-name. This folder is not either not existent "
                                   "or not included in this CAVE-object. Following folders are included in the analysis: %s" %
                                   (run, str(list(self.folder_to_run.keys()))))
                self.logger.debug("Using %s as cave-instance", run)
                if not cave:
                    raise ValueError("Using budgets, but didn't initialize CAVE-instances per run. "
                                     "Please run your example with '--verbose_level DEBUG' and "
                                     "report this issue with the debug.log (output_dir/debug/debug.log) "
                                     "on https://github.com/automl/CAVE/issues")
            else:
                raise ValueError("You are using a configurator that uses budgets. Please specify one of the following "
                                 "runs as a 'run=' keyword-argument: %s" % (str(list(self.folder_to_run.keys()))),)
        else:
            # Use aggregated objects
            self.logger.debug("Using aggregated cave-instance")
            cave = self
        self.logger.debug("Args: %s, Kwargs: %s", str(args), str(kw))
        try:
            analyzer = f(self, cave, *args, **kw)
        except Exception as err:
            self.logger.exception(err)
            raise
        else:
            if self.show_jupyter:
                try:
                    analyzer.get_jupyter()
                except ImportError as err:
                    self.logger.debug(err)
                    self.logger.info("Assuming that jupyter is not installed. Disable for rest of report.")
                    self.show_jupyter = False
                    pass
            if d is not None:
                analyzer.get_html(d)
        self.build_website()
        return analyzer
    return wrap

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
                 seed: int=42,
                 show_jupyter: bool=True,
                 verbose_level: str='OFF'):
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
        self.set_verbosity(verbose_level.upper())
        self.logger.debug("Running CAVE version %s", v)
        self.show_jupyter = show_jupyter
        # Methods that are never per-run, because they are inter-run-analysis by nature
        self.always_aggregated = ['bohb_learning_curves']  # these function-names will always be aggregated

        for d in os.listdir():
            if d.startswith('run_1'):
                shutil.rmtree(d)

        self.verbose_level = verbose_level
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.use_budgets = use_budgets
        self.ta_exec_dir = ta_exec_dir
        self.file_format = file_format
        self.validation_format = validation_format
        self.validation_method = validation_method
        self.pimp_max_samples = pimp_max_samples
        self.fanova_pairwise = fanova_pairwise

        # To be set during execution (used for dependencies of analysis-methods)
        self.param_imp = OrderedDict()
        self.feature_imp = OrderedDict()
        self.evaluators = []

        self.feature_names = None

        self.bohb_result = None  # only relevant for bohb_result

        # Create output_dir if necessary
        self.logger.info("Saving results to '%s'", self.output_dir)
        if not os.path.exists(output_dir):
            self.logger.debug("Output-dir '%s' does not exist, creating", self.output_dir)
            os.makedirs(output_dir)

        if file_format == 'BOHB':
            if len(folders) != 1:
                raise ValueError("For file format BOHB you can only specify one folder.")
            self.use_budgets = True
            self.bohb_result, folders = HpBandSter2SMAC().convert(folders[0])
            file_format = 'SMAC3'

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
        self.folder_to_run = {os.path.basename(run.folder) : run for run in self.runs}

        # Use scenario of first run for general purposes (expecting they are all the same anyway!
        self.scenario = self.runs[0].solver.scenario
        scenario_sanity_check(self.scenario, self.logger)
        self.feature_names = self.get_feature_names()
        self.default = self.scenario.cs.get_default_configuration()

        # All runs that have been actually explored during optimization
        self.global_original_rh = None
        # All original runs + validated runs if available
        self.global_validated_rh = None
        # All validated runs + EPM-estimated for def and inc on all insts
        self.global_epm_rh = None
        self.pimp = None
        self.model = None

        if self.use_budgets:
            self._init_helper_budgets()
        else:
            self._init_helper_no_budgets()

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
        """
        Each run gets it's own CAVE-instance. This way, we can simply use the individual objects (runhistories,
        output-dirs, etc)
        """
        for run in self.runs:
            sub_sec = os.path.basename(run.folder)
            # Set paths for each budget individual to avoid path-conflicts
            sub_output_dir = os.path.join(self.output_dir, 'content', sub_sec)
            os.makedirs(sub_output_dir, exist_ok=True)
            run.cave = CAVE(folders=[run.folder],
                            output_dir=sub_output_dir,
                            ta_exec_dir=[run.ta_exec_dir],
                            file_format=run.file_format,
                            validation_format=run.validation_format,
                            validation_method=self.validation_method,
                            pimp_max_samples=self.pimp_max_samples,
                            fanova_pairwise=self.fanova_pairwise,
                            use_budgets=False,
                            seed=self.seed,
                            verbose_level=self.verbose_level)

    def _init_helper_no_budgets(self):
        """
        No budgets means using global, aggregated runhistories to analyze the Configurator's behaviour.
        Also it creates an EPM using all available information, since all runs are "equal".
        """
        self.global_original_rh = RunHistory(average_cost)
        self.global_validated_rh = RunHistory(average_cost)
        self.global_epm_rh = RunHistory(average_cost)
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
                    "Feature Analysis",
                    "BOHB Plot",
                    ]
        for h in headings:
            self.website[h] = OrderedDict()

        if self.use_budgets:
            # The individual configurator runs are not directory comparable and cannot be aggregated.
            # Nevertheless they need to be combined in one comprehensive report and some metrics are to be compared over
            # the individual runs.

            # Perform analysis for each run
            if self.bohb_result:
                self.bohb_learning_curves(d=self.website)
            for run in self.runs:
                sub_sec = os.path.basename(run.folder)
                for h in headings:
                    self.website[h][sub_sec] = OrderedDict()
                # Set paths for each budget individual to avoid path-conflicts
                sub_output_dir = os.path.join(self.output_dir, 'content', sub_sec)
                os.makedirs(sub_output_dir, exist_ok=True)
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
                self.overview_table(d=self.website["Meta Data"][sub_sec], run=sub_sec)
                self.compare_default_incumbent(d=self.website["Best Configuration"][sub_sec], run=sub_sec)
                self.performance_analysis(self.website["Performance Analysis"][sub_sec], sub_sec,
                                          performance, cdf, scatter, algo_footprint)
                self.parameter_importance(self.website["Parameter Importance"][sub_sec], sub_sec,
                                          ablation='ablation' in param_importance,
                                          fanova='fanova' in param_importance,
                                          forward_selection='forward_selection' in param_importance,
                                          lpi='lpi' in param_importance,
                                          pimp_sort_table_by=pimp_sort_table_by)
                self.configurators_behavior(self.website["Configurator's Behavior"][sub_sec], sub_sec,
                                            cost_over_time,
                                            cfp, cfp_max_plot, cfp_time_slider, cfp_number_quantiles,
                                            parallel_coordinates)
                if self.feature_names:
                    self.feature_analysis(self.website["Feature Analysis"][sub_sec], sub_sec,
                                          box_violin='box_violin' in feature_analysis,
                                          correlation='correlation' in feature_analysis,
                                          clustering='clustering' in feature_analysis,
                                          importance='importance' in feature_analysis)
        else:
            self.overview_table(d=self.website["Meta Data"], run=None)
            self.compare_default_incumbent(d=self.website["Best Configuration"], run=None)
            self.performance_analysis(self.website["Performance Analysis"], None,
                                      performance, cdf, scatter, algo_footprint)
            self.parameter_importance(self.website["Parameter Importance"], None,
                                      ablation='ablation' in param_importance,
                                      fanova='fanova' in param_importance,
                                      forward_selection='forward_selection' in param_importance,
                                      lpi='lpi' in param_importance,
                                      pimp_sort_table_by=pimp_sort_table_by)
            self.configurators_behavior(self.website["Configurator's Behavior"], None,
                                        cost_over_time,
                                        cfp, cfp_max_plot, cfp_time_slider, cfp_number_quantiles,
                                        parallel_coordinates)
            if self.feature_names:
                self.feature_analysis(self.website["Feature Analysis"], None,
                                      box_violin='box_violin' in feature_analysis,
                                      correlation='correlation' in feature_analysis,
                                      clustering='clustering' in feature_analysis,
                                      importance='importance' in feature_analysis)

        self.build_website()

        self.logger.info("CAVE finished. Report is located in %s",
                         os.path.join(self.output_dir, 'report.html'))

    @analyzer_type
    def overview_table(self, cave):
        return OverviewTable(cave.scenario,
                             cave.global_original_rh,
                             cave.runs[0],
                             len(cave.runs),
                             cave.default,
                             cave.incumbent,
                             cave.output_dir)

    @analyzer_type
    def compare_default_incumbent(self, cave):
        return CompareDefaultIncumbent(cave.default, cave.incumbent)

    def performance_analysis(self, d, run,
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
            self.performance_table(d=d, run=run)
        if cdf:
            self.plot_ecdf(d=d, run=run)
        if scatter:
            self.plot_scatter(d=d, run=run)
        if algo_footprint and self.scenario.feature_dict:
            self.algorithm_footprints(d=d, run=run)

        self.build_website()

    @analyzer_type
    def performance_table(self, cave):
        instances = [i for i in cave.scenario.train_insts + cave.scenario.test_insts if i]
        return PerformanceTable(instances, cave.global_validated_rh, cave.default, cave.incumbent,
                                  cave.global_epm_rh, cave.scenario, cave.rng)

    @analyzer_type
    def plot_scatter(self, cave):
        return PlotScatter(default=cave.default,
                           incumbent=cave.incumbent,
                           rh=cave.global_epm_rh,
                           train=cave.scenario.train_insts,
                           test=cave.scenario.test_insts,
                           run_obj=cave.scenario.run_obj,
                           cutoff=cave.scenario.cutoff,
                           output_dir=cave.output_dir,
                           )

    @analyzer_type
    def plot_ecdf(self, cave):
        return PlotECDF(cave.default, cave.incumbent, cave.global_epm_rh,
                        cave.scenario.train_insts, cave.scenario.test_insts, cave.scenario.cutoff,
                        cave.output_dir)


    @analyzer_type
    def algorithm_footprints(self, cave):
        return AlgorithmFootprint(algorithms=[(cave.default, "default"), (cave.incumbent, "incumbent")],
                                  epm_rh=cave.global_epm_rh,
                                  train=cave.scenario.train_insts,
                                  test=cave.scenario.test_insts,
                                  features=cave.scenario.feature_dict,
                                  cutoff=cave.scenario.cutoff,
                                  output_dir=cave.output_dir,
                                  rng=cave.rng,
                                  )

    @analyzer_type
    def cost_over_time(self, cave):
        return CostOverTime(cave.scenario, cave.output_dir, cave.global_validated_rh, cave.runs, validator=cave.validator)

    @analyzer_type
    def parallel_coordinates(self, cave,
                             params: Union[int, List[str]]=10,
                             n_configs: int=100,
                             max_runs_epm: int=300000):
        """
        Plot parallel coordinates (visualize higher dimensions), here used
        to visualize the explored parameter configuration space.

        NOTE: the given runhistory should contain only optimization and no
        validation to analyze the explored parameter-space.

        Parameters
        ----------
        params: List[str] or int
            if int, plot at most params parameters, trying to determine with parameter importance.
            if List of strings, the names of the parameters to be plotted
        n_configs: int
            number of configs. will try to find most interesting configs to plot
        max_runs_epm: int
            this is a maximum of runs to be used for training of the epm. use to avoid MemoryErrors
        """
        self.logger.info("    plotting %s parameters for (max) %s configurations", params if isinstance(params, int)
                                                                                   else len(params), n_configs)

        return ParallelCoordinates(original_rh=cave.global_original_rh,
                                   validated_rh=cave.global_validated_rh,
                                   validator=cave.validator,
                                   scenario=cave.scenario,
                                   default=cave.default, incumbent=cave.incumbent,
                                   param_imp=cave.param_imp,
                                   params=params,
                                   max_runs_epm=max_runs_epm,
                                   output_dir=cave.output_dir,
                                   cs=cave.scenario.cs,
                                   runtime=(cave.scenario.run_obj == 'runtime'))

    @analyzer_type
    def configurator_footprint(self, cave,
                               time_slider=False, max_confs=1000, num_quantiles=8):
        self.logger.info("... visualizing explored configspace (this may take "
                         "a long time, if there is a lot of data - deactive with --no_configurator_footprint)")

        return ConfiguratorFootprint(
                 cave.scenario,
                 cave.runs,
                 cave.global_original_rh,
                 output_dir=cave.output_dir,
                 max_confs=max_confs,
                 time_slider=time_slider,
                 num_quantiles=num_quantiles)

    def configurators_behavior(self,
                               d,
                               run,
                               cost_over_time=False,
                               cfp=False,
                               cfp_max_plot=-1,
                               cfp_time_slider=False,
                               cfp_number_quantiles=1,
                               parallel_coordinates=False):

        if cost_over_time:
            self.cost_over_time(d=d, run=run)

        if cfp:  # Configurator Footprint
            self.configurator_footprint(d=d, run=run,
                                        time_slider=cfp_time_slider, max_confs=cfp_max_plot, num_quantiles=cfp_number_quantiles)

        if parallel_coordinates:
            # Should be after parameter importance, if performed.
            self.parallel_coordinates(d=d, run=run)

    @analyzer_type
    def cave_fanova(self, cave):
        fanova = CaveFanova(cave.pimp, cave.incumbent, cave.output_dir)
        cave.evaluators.append(cave.pimp.evaluator)
        cave.param_imp["fanova"] = cave.pimp.evaluator.evaluated_parameter_importance

        return fanova

    @analyzer_type
    def cave_ablation(self, cave):
        ablation = CaveAblation(cave.pimp, cave.incumbent, cave.output_dir)
        cave.evaluators.append(cave.pimp.evaluator)
        cave.param_imp["ablation"] = cave.pimp.evaluator.evaluated_parameter_importance

        return ablation

    @analyzer_type
    def pimp_forward_selection(self, cave):
        forward = CaveForwardSelection(cave.pimp, cave.incumbent, cave.output_dir)
        cave.evaluators.append(cave.pimp.evaluator)
        cave.param_imp["forward-selection"] = cave.pimp.evaluator.evaluated_parameter_importance

        return forward

    @analyzer_type
    def local_parameter_importance(self, cave):
        lpi = LocalParameterImportance(cave.pimp, cave.incumbent, cave.output_dir)
        cave.evaluators.append(cave.pimp.evaluator)
        cave.param_imp["lpi"] = cave.pimp.evaluator.evaluated_parameter_importance

        return lpi

    @analyzer_type
    def pimp_comparison_table(self, cave,
                              pimp_sort_table_by="average"):
        return PimpComparisonTable(cave.pimp,
                                   cave.evaluators,
                                   sort_table_by=pimp_sort_table_by,
                                   cs=cave.scenario.cs,
                                   out_fn=os.path.join(cave.output_dir, 'pimp.tex'),
                                   )

    def parameter_importance(self,
                             d, run,
                             ablation=False, fanova=False,
                             forward_selection=False, lpi=False, pimp_sort_table_by='average'):
        """Perform the specified parameter importance procedures. """
        sum_ = 0
        if fanova:
            self.cave_fanova(d=d, run=run)
            sum_ += 1
            self.logger.info("fANOVA...")

            self.build_website()

        if ablation:
            sum_ += 1
            self.logger.info("Ablation...")
            self.cave_ablation(d=d, run=run)
            self.build_website()

        if forward_selection:
            sum_ += 1
            self.pimp_forward_selection(d=d, run=run)
            self.logger.info("Forward Selection...")
            self.build_website()

        if lpi:
            sum_ += 1
            self.logger.info("Local EPM-predictions around incumbent...")
            self.local_parameter_importance(d=d, run=run)
            self.build_website()

        if sum_ >= 2:
            self.pimp_comparison_table(d=d, run=run)
            self.build_website()

    @analyzer_type
    def feature_importance(self, cave):
        res = FeatureImportance(cave.pimp, cave.output_dir)
        cave.feature_imp = res.feat_importance
        return res

    @analyzer_type
    def box_violin(self, cave):
        return BoxViolin(cave.output_dir,
                         cave.scenario,
                         cave.feature_names,
                         cave.feature_imp)


    @analyzer_type
    def feature_correlation(self, cave):
        return FeatureCorrelation(cave.output_dir,
                                cave.scenario,
                                cave.feature_names,
                                cave.feature_imp)


    @analyzer_type
    def feature_clustering(self, cave):
        return FeatureClustering(cave.output_dir,
                                 cave.scenario,
                                 cave.feature_names,
                                 cave.feature_imp)

    def feature_analysis(self, d, run,
                         box_violin=False, correlation=False, clustering=False, importance=False):
        # feature importance using forward selection
        if importance:
            self.feature_importance(d=d, run=run)
        if box_violin:
            self.box_violin(d=d, run=run)
        if correlation:
            self.feature_correlation(d=d, run=run)
        if clustering:
            self.feature_clustering(d=d, run=run)
        self.build_website()

    @analyzer_type
    def bohb_learning_curves(self, cave):
        return BohbLearningCurves(self.scenario.cs.get_hyperparameter_names(), result_object=self.bohb_result)

############################################################################
############################################################################
    def print_budgets(self):
        if self.use_budgets:
            print(list(self.folder_to_run.keys()))
        else:
            print("This cave instance does not seem to use budgets.")

    def get_feature_names(self):
        if not self.scenario.feature_dict:
            self.logger.info("No features available. Skipping feature analysis.")
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
        return feat_names

    def build_website(self):
        self.builder.generate_html(self.website)

    def set_verbosity(self, level):
        # Log to stream (console)
        logging.getLogger().setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
        stdout_handler = logging.StreamHandler()
        stdout_handler.setFormatter(formatter)
        if level == "INFO":
            stdout_handler.setLevel(logging.INFO)
        elif level in ["OFF", "WARNING"]:
            stdout_handler.setLevel(logging.WARNING)
        elif level in ["DEBUG", "DEV_DEBUG"]:
            stdout_handler.setLevel(logging.DEBUG)
            if level == "DEV_DEBUG":
                # Disable annoying boilerplate-debug-logs from foreign modules
                disable_loggers = ["smac.scenario",
                                   # pimp logging
                                   "pimp.epm.unlogged_epar_x_rfwi.UnloggedEPARXrfi",
                                   "Forward-Selection",
                                   "LPI",
                                   # Other (mostly bokeh)
                                   "PIL.PngImagePlugin",
                                   "matplotlib.font_manager",
                                   "urllib3.connectionpool",
                                   "selenium.webdriver.remote.remote_connection"]
                for logger in disable_loggers:
                    logging.getLogger().debug("Setting logger \'%s\' on level INFO", logger)
                    logging.getLogger(logger).setLevel(logging.INFO)
        else:
            raise ValueError("%s not recognized as a verbosity level. Choose from DEBUG, DEV_DEBUG. INFO, OFF.".format(level))

        logging.getLogger().addHandler(stdout_handler)
        # Log to file
        if not os.path.exists(os.path.join(self.output_dir, "debug")):
            os.makedirs(os.path.join(self.output_dir, "debug"))
        fh = logging.FileHandler(os.path.join(self.output_dir, "debug/debug.log"), "w")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logging.getLogger().addHandler(fh)
