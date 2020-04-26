import logging
import os
import shutil
import tempfile
import typing
from collections import OrderedDict
from functools import wraps
from importlib import reload
from typing import Union, List

import numpy as np

from cave.__version__ import __version__ as v
from cave.analyzer.algorithm_footprint import AlgorithmFootprint
from cave.analyzer.bohb_incumbents_per_budget import BohbIncumbentsPerBudget
from cave.analyzer.bohb_learning_curves import BohbLearningCurves
from cave.analyzer.box_violin import BoxViolin
from cave.analyzer.budget_correlation import BudgetCorrelation
from cave.analyzer.cave_ablation import CaveAblation
from cave.analyzer.cave_fanova import CaveFanova
from cave.analyzer.cave_forward_selection import CaveForwardSelection
from cave.analyzer.compare_default_incumbent import CompareDefaultIncumbent
from cave.analyzer.configurator_footprint import ConfiguratorFootprint
from cave.analyzer.cost_over_time import CostOverTime
from cave.analyzer.feature_clustering import FeatureClustering
from cave.analyzer.feature_correlation import FeatureCorrelation
from cave.analyzer.feature_importance import FeatureImportance
from cave.analyzer.local_parameter_importance import LocalParameterImportance
from cave.analyzer.overview_table import OverviewTable
from cave.analyzer.parallel_coordinates import ParallelCoordinates
from cave.analyzer.performance_table import PerformanceTable
from cave.analyzer.pimp_comparison_table import PimpComparisonTable
from cave.analyzer.plot_ecdf import PlotECDF
from cave.analyzer.plot_scatter import PlotScatter
from cave.html.html_builder import HTMLBuilder
from cave.reader.runs_container import RunsContainer
from cave.utils.exceptions import Deactivated, NotApplicable
from cave.utils.helpers import load_default_options
from cave.utils.timing import timing

__author__ = "Joshua Marben"
__copyright__ = "Copyright 2017, ML4AAD"
__license__ = "3-clause BSD"
__maintainer__ = "Joshua Marben"
__email__ = "joshua.marben@neptun.uni-freiburg.de"


def _analyzer_type(f):
    @wraps(f)
    def wrap(self, *args, d=None, **kw):
        self.logger.debug("Args: %s, Kwargs: %s", str(args), str(kw))
        try:
            analyzer = f(self, *args, **kw)
        except Deactivated as err:
            self.logger.info(err)
            return
        except NotApplicable as err:
            self.logger.debug("Skipping analyzer ({})".format(err))
            return
        except Exception as err:
            self.logger.exception(err)
            raise
        else:
            # execute hooks, if possible and/or desired
            if self.show_jupyter:
                try:
                    analyzer.get_jupyter()
                except ImportError as err:
                    self.logger.debug(err)
                    self.logger.info("Assuming that jupyter is not installed. Disable for rest of report.")
                    self.show_jupyter = False
            if isinstance(d, dict):
                analyzer.get_html(d)
        self._build_website()
        return analyzer
    return wrap

class CAVE(object):
    def __init__(self,
                 folders: typing.List[str],
                 output_dir: str,
                 ta_exec_dir: typing.List[str],
                 file_format: str='auto',
                 validation_format='NONE',
                 validation_method: str='epm',
                 seed: int=42,
                 show_jupyter: bool=True,
                 verbose_level: str='OFF',
                 analyzing_options=None,
                 **kwargs
                 ):
        """
        Initialize CAVE facade to handle analyzing, plotting and building the report-page easily.
        During initialization, the analysis-infrastructure is built and the data is validated, the overall best
        incumbent is found and default+incumbent are evaluated for all instances for all runs, by default using an EPM.

        Arguments
        ---------
        folders: list<strings>
            paths to relevant SMAC runs
        output_dir: string
            output for cave to write results (figures + report)
        ta_exec_dir: string
            execution directory for target algorithm (to find instance.txt specified in scenario, ..)
        file_format: str
            what format the rundata is in, options are [SMAC3, SMAC2, BOHB and CSV]
        validation_format: str
            what format the validation rundata is in, options are [SMAC3, SMAC2, CSV and None]
        validation_method: string
            from [validation, epm], how to estimate missing runs
        seed: int
            random seed for analysis (e.g. the random forests)
        show_jupyter: bool
            default True, tries to output plots and tables to jupyter-frontend, if available
        verbose_level: str
            from [OFF, INFO, DEBUG, DEV_DEBUG and WARNING]
        analyzing_options: str or dict
            options-dictionary following CAVE's options-syntax
        """
        self.show_jupyter = show_jupyter
        if self.show_jupyter:
            # Reset logging module (needs to happen before logger initalization)
            logging.shutdown()
            reload(logging)

        self.logger = logging.getLogger(self.__module__ + '.' + self.__class__.__name__)
        self.output_dir = output_dir
        self.output_dir_created = False  # this flag avoids multiple output-dir creations

        # Create output_dir and set verbosity
        self.set_verbosity(verbose_level.upper())
        self._create_outputdir(self.output_dir)

        self.logger.debug("Running CAVE version %s", v)

        self.verbose_level = verbose_level
        self.rng = np.random.RandomState(seed)
        self.folders = folders
        self.ta_exec_dir = ta_exec_dir
        self.file_format = file_format
        self.validation_format = validation_format
        self.validation_method = validation_method

        # Configuration of analyzers (works as a default for report generation)
        analyzing_options = load_default_options(analyzing_options, file_format)

        self.runscontainer = RunsContainer(folders=self.folders,
                                           ta_exec_dirs=self.ta_exec_dir,
                                           output_dir=self.output_dir,
                                           file_format=self.file_format,  # TODO remove?
                                           validation_format=self.validation_format,  # TODO remove?
                                           analyzing_options=analyzing_options,
                                           )

        # create builder for html-website, decide for suitable logo
        custom_logo = './custom_logo.png'
        if self.runscontainer.file_format == 'BOHB':
            logo_fn = 'BOHB_logo.png'
        elif self.runscontainer.file_format.startswith('SMAC'):
            logo_fn = 'SMAC_logo.png'
        elif os.path.exists(custom_logo):
            logo_fn = custom_logo
        else:
            logo_fn = 'automl-logo.png'
            self.logger.info("No suitable logo found. You can use a custom logo simply by having a file called '%s' "
                             "in the directory from which you run CAVE.", custom_logo)
        self.builder = HTMLBuilder(self.output_dir, "CAVE", logo_fn=logo_fn, logo_custom=custom_logo==logo_fn)
        self.website = OrderedDict([])

    @timing
    def analyze(self,
                options=None):
        """Analyze the available data and build HTML-webpage as dict.
        Save webpage in 'self.output_dir/CAVE/report.html'.

        Parameters
        ----------
        options: Dict or str
            either a dictionary or a path to an ini-file.
        """
        flag_show_jupyter = self.show_jupyter
        self.show_jupyter = False

        # Process analyzing-options
        if isinstance(options, str):
            self.logger.debug("Loading \"{}\".".format(options))
            self.runscontainer.analyzing_options.read(options)
        elif isinstance(options, dict):
            for k, v in options.items():
                occurences = sum([1 for x in self.runscontainer.analyzing_options.sections() if k in x])
                if occurences > 1:
                    self.logger.warning("{} is an arbitrary option - to avoid collisions, "
                                        "consider using a different name.")
                if occurences == 0 and k not in self.runscontainer.analyzing_options.sections():
                    self.logger.warning("{} is not found in default options. Are you sure you know what you are doing?")
                for s in self.runscontainer.analyzing_options.sections():
                    if s == k:
                        self.runscontainer.analyzing_options[k]['run'] = str(v)
                    elif k in self.runscontainer.analyzing_options[s]:
                        self.runscontainer.analyzing_options[s][k] = str(v)

        self.overview_table(d=self.website)
        self.compare_default_incumbent(d=self._get_dict(self.website, "Meta Data"))

        ###################################################
        #  Performance Analysis  #  Performance Analysis  #
        ###################################################
        title = "Performance Analysis"
        self.performance_table(d=self._get_dict(self.website, title))
        self.plot_ecdf(d=self._get_dict(self.website, title))
        self.plot_scatter(d=self._get_dict(self.website, title))
        self.algorithm_footprints(d=self._get_dict(self.website, title))

        ###################################################
        #    Budget  Analysis    #    Budget  Analysis    #
        ###################################################
        title = "Budget Analysis"
        if self.runscontainer.get_budgets() is not None:
            self.bohb_incumbents_per_budget(d=self._get_dict(self.website, title))
            self.budget_correlation(d=self._get_dict(self.website, title))
            if self.runscontainer.file_format == "BOHB":
                self.bohb_learning_curves(d=self._get_dict(self.website, title))

        ###################################################
        #         Parameter- and Feature-Analysis         #
        ###################################################
        self.parameter_importance(self._get_dict(self.website, "Parameter Importance"))
        self.feature_analysis(self._get_dict(self.website, "Feature Analysis"))

        ###################################################
        #            Configurator's Behaviour             #
        ###################################################
        title = "Configurators Behavior"
        self.configurator_footprint(d=self._get_dict(self.website, title))
        self.cost_over_time(d=self._get_dict(self.website, title))
        # Parallel Coordinates should be after parameter importance, if performed.
        self.parallel_coordinates(d=self._get_dict(self.website, title))

        self._build_website()

        self.logger.info("CAVE finished. Report is located in %s",
                         os.path.join(self.output_dir, 'report.html'))

        self.show_jupyter = flag_show_jupyter

    def _get_dict(self, d, layername):
        """ Get the appropriate sub-dict for this layer (or layer-run combination) and create it if necessary """
        if not isinstance(d, dict):
            raise ValueError("Pass a valid dict to _get_dict!")
        if not layername in d:
            d[layername] = OrderedDict()
        return d[layername]

    @_analyzer_type
    def overview_table(self):
        return OverviewTable(self.runscontainer)

    @_analyzer_type
    def compare_default_incumbent(self):
        return CompareDefaultIncumbent(self.runscontainer)

    @_analyzer_type
    def performance_table(self):
        return PerformanceTable(self.runscontainer)

    @_analyzer_type
    def plot_scatter(self):
        return PlotScatter(self.runscontainer)

    @_analyzer_type
    def plot_ecdf(self):
        return PlotECDF(self.runscontainer)

    @_analyzer_type
    def algorithm_footprints(self):
        return AlgorithmFootprint(self.runscontainer)

    @_analyzer_type
    def cost_over_time(self,
                       incumbent_trajectory=None):
        return CostOverTime(self.runscontainer,
                            incumbent_trajectory=incumbent_trajectory)

    @_analyzer_type
    def parallel_coordinates(self,
                             pc_sort_by: str=None,
                             params: Union[int, List[str]]=None,
                             n_configs: int=None,
                             max_runs_epm: int=None,
                             ):
        return ParallelCoordinates(self.runscontainer,
                                   pc_sort_by=pc_sort_by,
                                   params=params,
                                   n_configs=n_configs,
                                   max_runs_epm=max_runs_epm,
                                   )

    @_analyzer_type
    def configurator_footprint(self,
                               time_slider=None,
                               max_configurations_to_plot=None,
                               number_quantiles=None):
        return ConfiguratorFootprint(self.runscontainer,
                                     time_slider=time_slider,
                                     max_configurations_to_plot=max_configurations_to_plot,
                                     number_quantiles=number_quantiles)

    @_analyzer_type
    def cave_fanova(self):
        try:
            fanova = CaveFanova(self.runscontainer)

        except IndexError as err:
            self.logger.debug("Error in fANOVA (%s)", err, exc_info=1)
            raise IndexError("Error in fANOVA - please run with --pimp_no_fanova_pairs (this is due to a known issue "
                             "with ints and bools in categorical hyperparameters, see issue #192).")
        return fanova

    @_analyzer_type
    def cave_ablation(self):
        return CaveAblation(self.runscontainer)

    @_analyzer_type
    def pimp_forward_selection(self):
        return CaveForwardSelection(self.runscontainer)

    @_analyzer_type
    def local_parameter_importance(self):
        return LocalParameterImportance(self.runscontainer)

    @_analyzer_type
    def pimp_comparison_table(self,
                              pimp_sort_table_by="average"):
        return PimpComparisonTable(self.runscontainer,
                                   sort_table_by=pimp_sort_table_by,
                                   )

    def parameter_importance(self,
                             d):
        """Perform the specified parameter importance procedures. """
        self.cave_fanova(d=d)
        self.cave_ablation(d=d)
        self.pimp_forward_selection(d=d)
        self.local_parameter_importance(d=d)
        if sum([1 for x in ['fANOVA', 'Ablation', 'Forward Selection', 'Local Parameter Importance (LPI)'] if self.runscontainer.analyzing_options[x]['run'] != 'False']) >= 2:
            pct = self.pimp_comparison_table(d=d)
            d.move_to_end(pct.name, last=False)

    @_analyzer_type
    def feature_importance(self):
        res = FeatureImportance(self.runscontainer)
        return res

    @_analyzer_type
    def box_violin(self):
        return BoxViolin(self.runscontainer)


    @_analyzer_type
    def feature_correlation(self):
        return FeatureCorrelation(self.runscontainer)


    @_analyzer_type
    def feature_clustering(self):
        return FeatureClustering(self.runscontainer)

    def feature_analysis(self, d):
        # feature importance using forward selection
        self.feature_importance(d=d)
        self.box_violin(d=d)
        self.feature_correlation(d=d)
        self.feature_clustering(d=d)

    @_analyzer_type
    def bohb_learning_curves(self):
        return BohbLearningCurves(self.runscontainer)

    @_analyzer_type
    def bohb_incumbents_per_budget(self):
        return BohbIncumbentsPerBudget(self.runscontainer)

    @_analyzer_type
    def budget_correlation(self):
        return BudgetCorrelation(self.runscontainer)


###########################################################################
# HELPERS HELPERS HELPERS HELPERS HELPERS HELPERS HELPERS HELPERS HELPERS #
###########################################################################

    def print_budgets(self):
        """If the analyzed configurator uses budgets, print a list of available budgets."""
        print(self.runscontainer.get_budgets())

    def _build_website(self):
        self.builder.generate_webpage(self.website)

    def set_verbosity(self, level):
        logging.getLogger().setLevel(logging.DEBUG)
        # Log to stream (console)
        formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
        stdout_handler = logging.StreamHandler()
        stdout_handler.setFormatter(formatter)
        if level == "INFO":
            stdout_handler.setLevel(logging.INFO)
        elif level == "WARNING":
            stdout_handler.setLevel(logging.WARNING)
        elif level == "OFF":
            stdout_handler.setLevel(logging.ERROR)
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
                                   "matplotlib",
                                   "urllib3.connectionpool",
                                   "selenium.webdriver.remote.remote_connection"]
                for logger in disable_loggers:
                    logging.getLogger('cave.settings').debug("Setting logger \'%s\' on level INFO", logger)
                    logging.getLogger(logger).setLevel(logging.INFO)
        else:
            raise ValueError("%s not recognized as a verbosity level. Choose from DEBUG, DEV_DEBUG. INFO, WARNING, OFF.".format(level))

        logging.getLogger().addHandler(stdout_handler)

        # Log to file is always debug
        debug_path = os.path.join(self.output_dir, "debug", "debug.log")
        logging.getLogger('cave.settings').debug("Output-file for debug-log: '%s'", debug_path)
        self._create_outputdir(self.output_dir)
        os.makedirs(os.path.split(debug_path)[0])
        fh = logging.FileHandler(debug_path, "a")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logging.getLogger().addHandler(fh)
        
        self.logger.debug("Existing loggers: %s", str(logging.root.manager.loggerDict))

    def _create_outputdir(self, output_dir):
        """ Creates output-dir, if necessary. Also sets the 'self.output_dir_created'-flag, so this only happens once.
        If there is a directory already, zip this into an archive in the output_dir called '.OLD.zip'. """
        if self.output_dir_created:
            if not os.path.exists(output_dir):
                raise RuntimeError("'%s' should exist, but doesn't. Any raceconditions? "
                                   "Please report to github.com/automl/CAVE/issues with debug/debug.log")
            self.logger.debug("Output-dir '%s' was already created, call ignored", output_dir)
            return

        self.logger.info("Saving results to '%s'", output_dir)
        if not os.path.exists(output_dir):
            self.logger.debug("Output-dir '%s' does not exist, creating", output_dir)
            os.makedirs(output_dir)
        else:
            archive_path = shutil.make_archive(os.path.join(tempfile.mkdtemp(), '.OLD'), 'zip', output_dir)
            shutil.rmtree(output_dir)
            os.makedirs(output_dir)
            shutil.move(archive_path, output_dir)
            self.logger.debug("Output-dir '%s' exists, moving old content to '%s'", self.output_dir,
                              os.path.join(self.output_dir, '.OLD.zip'))

        self.output_dir_created = True
