import copy
import logging
import os
import tempfile
from collections import OrderedDict
from contextlib import contextmanager

import numpy as np
from pimp.importance.importance import Importance
from smac.runhistory.runhistory import RunHistory, DataOrigin
from smac.utils.io.input_reader import InputReader
from smac.utils.validate import Validator
from smac import __version__ as smac_version

from cave.reader.smac2_reader import SMAC2Reader
from cave.reader.smac3_reader import SMAC3Reader
from cave.utils.helpers import scenario_sanity_check
from cave.utils.timing import timing


class ConfiguratorRun(object):
    """
    ConfiguratorRuns load and maintain information about individual configurator
    runs. There are different supported formats, like: BOHB, SMAC3, SMAC2 and CSV
    This class is responsible for providing a scenario, a runhistory and a
    trajectory and handling original/validated data appropriately.
    To create a ConfiguratorRun from a folder, use Configurator.from_folder()
    """
    def __init__(self,
                 scenario,
                 original_runhistory,
                 validated_runhistory,
                 trajectory,
                 options,
                 path_to_folder=None,
                 ta_exec_dir=None,
                 file_format=None,
                 validation_format=None,
                 reduced_to_budgets=None,
                 output_dir=None,
                 ):
        """
        Parameters
        ----------
        scenario: Scenario
            scenario
        original_runhistory, validated_runhistory: RunHistory
            runhistores containing only the original evaluated data (during optimization process) or the validated data
            where points of interest are reevaluated after the optimization process
        trajectory: List[dict]
            a trajectory of the best performing configurations at each point in time
        options: dict
            options can define a number of custom settings
        path_to_folder: str
            path to the physical folder containing the data
        ta_exec_dir: str
            path to the target-algorithm-execution-directory. This is only important for SMAC-optimized data
        file_format, validation_format: str
            will be autodetected some point soon, until then, specify the file-format (SMAC2, SMAC3, BOHB, etc...)
        reduced_to_budgets: List str int or float
            budgets, with which this cr is associated
        output_dir: str
            where to save analysis-data for this cr
        """
        self.logger = logging.getLogger("cave.ConfiguratorRun.{}".format(path_to_folder))
        self.rng = np.random.RandomState(42)
        self.options = options

        self.path_to_folder = path_to_folder
        self.reduced_to_budgets = [None] if reduced_to_budgets is None else reduced_to_budgets

        self.scenario = scenario
        self.original_runhistory = original_runhistory
        self.validated_runhistory = validated_runhistory
        self.trajectory = trajectory
        self.ta_exec_dir = ta_exec_dir
        self.file_format = file_format
        self.validation_format = validation_format
        if not output_dir:
            self.logger.debug("New outputdir")
            output_dir = tempfile.mkdtemp()
        self.output_dir = os.path.join(output_dir, 'analysis_data', self.get_identifier())
        os.makedirs(self.output_dir, exist_ok=True)

        self.default = self.scenario.cs.get_default_configuration()
        self.incumbent = self.trajectory[-1]['incumbent'] if self.trajectory else None
        self.feature_names = self._get_feature_names()

        # Create combined runhistory to collect all "real" runs
        self.combined_runhistory = RunHistory()
        self.combined_runhistory.update(self.original_runhistory, origin=DataOrigin.INTERNAL)
        if self.validated_runhistory is not None:
            self.combined_runhistory.update(self.validated_runhistory, origin=DataOrigin.EXTERNAL_SAME_INSTANCES)

        # Create runhistory with estimated runs (create Importance-object of pimp and use epm-model for validation)
        self.epm_runhistory = RunHistory()
        self.epm_runhistory.update(self.combined_runhistory)

        # Initialize importance and validator
        self._init_pimp_and_validator()
        try:
            self._validate_default_and_incumbents("epm", self.ta_exec_dir)
        except KeyError as err:
            self.logger.debug(err, exc_info=1)
            msg = 'Validation of default and incumbent failed. SMAC (v: %s) does not support validation '\
                  'of budgets+instances yet, if you use budgets but no instances ignore this warning.' % str(smac_version)
            if self.feature_names:
                self.logger.warning(msg)
            else:
                self.logger.debug(msg)

        # Set during execution, to share information between Analyzers
        self.share_information = {'parameter_importance' : OrderedDict(),
                                  'feature_importance' : OrderedDict(),
                                  'evaluators' : OrderedDict(),
                                  'validator' : None,
                                  'hpbandster_result' : None,  # Only for file-format BOHB
                                  }

    def get_identifier(self):
        return self.identify(self.path_to_folder, self.reduced_to_budgets)

    @classmethod
    def identify(cls, path, budget):
        path = path if path is not None else "all_folders"
        budget = str(budget) if budget is not None else "all_budgets"
        res = "_".join([path, budget]).replace('/', '_')
        if len(res) > len(str(hash(res))):
            res = str(hash(res))
        return res


    def get_budgets(self):
        return set([k.budget for k in self.original_runhistory.data.keys()])

    @classmethod
    def from_folder(cls,
                    folder: str,
                    ta_exec_dir: str,
                    options,
                    file_format: str='SMAC3',
                    validation_format: str='NONE',
                    output_dir=None,
                    ):
        """Initialize scenario, runhistory and incumbent from folder

        Parameters
        ----------
        folder: string
            output-dir of this configurator-run -> this is also the 'id' for a single run in parallel optimization
        ta_exec_dir: string
            if the execution directory for the SMAC-run differs from the cwd,
            there might be problems loading instance-, feature- or PCS-files
            in the scenario-object. since instance- and PCS-files are necessary,
            specify the path to the execution-dir of SMAC here
        file_format: string
            from [SMAC2, SMAC3, BOHB, CSV]
        validation_format: string
            from [SMAC2, SMAC3, CSV, NONE], in which format to look for validated data
        """
        logger = logging.getLogger("cave.ConfiguratorRun.{}".format(folder))
        logger.debug("Loading from \'%s\' with ta_exec_dir \'%s\' with file-format '%s' and validation-format %s. ",
                          folder, ta_exec_dir, file_format, validation_format)

        if file_format == 'BOHB':
            logger.debug("File format is BOHB, assmuming data was converted to SMAC3-format using "
                         "HpBandSter2SMAC from cave.reader.converter.hpbandster2smac.")
        validation_format = validation_format if validation_format != 'NONE' else None

        #### Read in data (scenario, runhistory & trajectory)
        reader = cls.get_reader(file_format, folder, ta_exec_dir)

        scenario = reader.get_scenario()
        scenario_sanity_check(scenario, logger)
        original_runhistory = reader.get_runhistory(scenario.cs)
        validated_runhistory = None

        if validation_format == "NONE" or validation_format is None:
            validation_format = None
        else:
            logger.debug('Using format %s for validation', validation_format)
            vali_reader = cls.get_reader(validation_format, folder, ta_exec_dir)
            vali_reader.scen = scenario
            validated_runhistory = vali_reader.get_validated_runhistory(scenario.cs)
            #self._check_rh_for_inc_and_def(self.validated_runhistory, 'validated runhistory')
            logger.info("Found validated runhistory for \"%s\" and using "
                         "it for evaluation. #configs in validated rh: %d",
                         folder, len(validated_runhistory.config_ids))

        trajectory = reader.get_trajectory(scenario.cs)

        return cls(scenario,
                   original_runhistory,
                   validated_runhistory,
                   trajectory,
                   options,
                   path_to_folder=folder,
                   ta_exec_dir=ta_exec_dir,
                   file_format=file_format,
                   validation_format=validation_format,
                   output_dir=output_dir,
                   )

    def get_incumbent(self):
        return self.incumbent

    def _init_pimp_and_validator(self,
                                 alternative_output_dir=None,
                                 ):
        """Create ParameterImportance-object and use it's trained model for validation and further predictions.
        We pass a combined (original + validated) runhistory, so that the returned model will be based on as much
        information as possible

        Parameters
        ----------
        alternative_output_dir: str
            e.g. for budgets we want pimp to use an alternative output-dir (subfolders per budget)
        """
        self.logger.debug("Using '%s' as output for pimp", alternative_output_dir if alternative_output_dir else
                                                           self.output_dir)
        self.pimp = Importance(scenario=copy.deepcopy(self.scenario),
                               runhistory=self.combined_runhistory,
                               incumbent=self.incumbent if self.incumbent else self.default,
                               save_folder=alternative_output_dir if alternative_output_dir is not None else self.output_dir,
                               seed=self.rng.randint(1, 100000),
                               max_sample_size=self.options['fANOVA'].getint("pimp_max_samples"),
                               fANOVA_pairwise=self.options['fANOVA'].getboolean("fanova_pairwise"),
                               preprocess=False,
                               verbose=False,  # disable progressbars in pimp...
                               )
        # Validator (initialize without trajectory)
        self.validator = Validator(self.scenario, None, None)
        self.validator.epm = self.pimp.model

    @timing
    def _validate_default_and_incumbents(self,
                                         method,
                                         ta_exec_dir,
                                         ):
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
        # TODO maybe just validate whole trajectory?
        self.logger.debug("Validating %s using %s!", self.get_identifier(), method)
        self.validator.traj = self.trajectory
        if method == "validation":
            with _changedir(ta_exec_dir):
                # TODO determine # repetitions
                new_rh = self.validator.validate('def+inc', 'train+test', 1, -1, runhistory=self.combined_runhistory)
            self.validated_runhistory.update(new_rh)
            self.combined_runhistory_rh.update(new_rh)
        elif method == "epm":
            # Only do test-instances if features for test-instances are available
            instance_mode = 'train+test'
            if (any([i not in self.scenario.feature_dict for i in self.scenario.test_insts]) and
                any([i in self.scenario.feature_dict for i in self.scenario.train_insts])):  # noqa
                self.logger.debug("No features provided for test-instances (but for train!). Cannot validate on \"epm\".")
                self.logger.warning("Features detected for train-instances, but not for test-instances. This is "
                                    "unintended usage and may lead to errors for some analysis-methods.")
                instance_mode = 'train'

            new_rh = self.validator.validate_epm('def+inc', instance_mode, 1, runhistory=self.combined_runhistory)
            self.epm_runhistory.update(new_rh)
        else:
            raise ValueError("Missing data method illegal (%s)", method)
        self.validator.traj = None  # Avoid usage-mistakes

    def _get_feature_names(self):
        if not self.scenario.feature_dict:
            self.logger.info("No features available. Skipping feature analysis.")
            return
        feat_fn = self.scenario.feature_fn
        if not self.scenario.feature_names:
            self.logger.debug("`scenario.feature_names` is not set. Loading from '%s'", feat_fn)
            with _changedir(self.ta_exec_dir if self.ta_exec_dir else '.'):
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

    def _check_rh_for_inc_and_def(self, rh, name=''):
        """
        Check if default and incumbent are evaluated on all instances in this rh

        Parameters
        ----------
        rh: RunHistory
            runhistory to be checked
        name: str
            name for logging-purposes

        Returns
        -------
        return_value: bool
            False if either inc or def was not evaluated on all
            train/test-instances
        """
        return_value = True
        for c_name, c in [("default", self.default), ("inc", self.incumbent)]:
            runs = rh.get_runs_for_config(c, only_max_observed_budget=False)
            evaluated = set([inst for inst, seed in runs])
            for i_name, i in [("train", self.train_inst),
                              ("test", self.test_inst)]:
                not_evaluated = set(i) - evaluated
                if len(not_evaluated) > 0:
                    self.logger.debug("RunHistory %s only evaluated on %d/%d %s-insts for %s in folder %s",
                                      name, len(i) - len(not_evaluated), len(i), i_name, c_name, self.folder)
                    return_value = False
        return return_value

    @classmethod
    def get_reader(cls, name, folder, ta_exec_dir):
        """ Returns an appropriate reader for the specified format. """
        # TODO make autodetect format (here? where?)
        if name == 'SMAC3':
            return SMAC3Reader(folder, ta_exec_dir)
        elif name == 'BOHB':
            return SMAC3Reader(folder, ta_exec_dir)
        elif name == 'SMAC2':
            return SMAC2Reader(folder, ta_exec_dir)
        elif name == 'CSV':
            return SMAC3Reader(folder, ta_exec_dir)
        else:
            raise ValueError("%s not supported as file-format" % name)

@contextmanager
def _changedir(newdir):
    """ Helper function to change directory, for example to create a scenario from file, where paths to the instance-
    and feature-files are relative to the original SMAC-execution-directory. Same with target algorithms that need
    be executed for validation. """
    olddir = os.getcwd()
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(olddir)
