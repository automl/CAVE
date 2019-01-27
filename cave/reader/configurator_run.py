import logging

from smac.facade.smac_facade import SMAC
from smac.optimizer.objective import average_cost
from smac.runhistory.runhistory import RunHistory, DataOrigin

from cave.reader.smac3_reader import SMAC3Reader
from cave.reader.smac2_reader import SMAC2Reader
from cave.reader.csv_reader import CSVReader

def get_reader(name):
    """ Returns an appropriate reader for the specified format. """
    if name == 'SMAC3':
        return SMAC3Reader(folder, ta_exec_dir)
    elif name == 'BOHB':
        self.logger.debug("File format is BOHB, assmuming data was converted to SMAC3-format using "
                          "HpBandSter2SMAC from cave.utils.converter.hpbandster2smac.")
        return SMAC3Reader(folder, ta_exec_dir)
    elif name == 'SMAC2':
        return SMAC2Reader(folder, ta_exec_dir)
    elif name == 'CSV':
        return CSVReader(folder, ta_exec_dir)
    else:
        raise ValueError("%s not supported as file-format" % name)

class ConfiguratorRun(SMAC):
    """
    ConfiguratorRuns load and maintain information about individual configurator
    runs. There are three supported formats: SMAC3, SMAC2 and CSV
    This class is responsible for providing a scenario, a runhistory and a
    trajectory and handling original/validated data appropriately.
    """
    def __init__(self,
                 scenario,
                 original_runhistory,
                 validated_runhistory,
                 trajectory,
                 folder,
                 ta_exec_dir,
                 file_format,
                 validation_format,
                 budget=None,
                 ):
        self.scenario = scenario
        self.original_runhistory = original_runhistory
        self.validated_runhistory = validated_runhistory
        self.trajectory = trajectory
        self.path_to_folder = path_to_folder
        self.ta_exec_dir = ta_exec_dir
        self.file_format = file_format
        self.validation_format = validation_format
        self.budget = budget

        self.default = self.scenario.cs.get_default_configuration()
        self.incumbent = self.trajectory[-1]['incumbent'] if self.trajectory else None
        self.feature_names = self._get_feature_names()

        # Create combined runhistory to collect all "real" runs
        self.combined_runhistory = RunHistory(average_cost)
        self.combined_runhistory.update(self.original_runhistory, origin=DataOrigin.INTERNAL)
        if self.validated_runhistory:
            self.combined_runhistory.update(self.validated_runhistory, origin=DataOrigin.EXTERNAL_SAME_INSTANCES)

        # Create runhistory with estimated runs (create Importance-object of pimp and use epm-model for validation)
        self.epm_runhistory = RunHistory(average_cost)
        self.epm_runhistory.update(self.combined_runhistory)
        self._init_pimp_and_validator()


        # Set during execution, to share information between Analyzers
        self.share_information = {'parameter_importance' : OrderedDict(),
                                  'feature_importance' : OrderedDict(),
                                  'evaluators' : [],
                                  'validator' : None}

        # Initialize SMAC-object
        super().__init__(scenario=self.scen, runhistory=self.combined_runhistory)  # restore_incumbent=incumbent)
        # TODO use restore, delete next line
        self.solver.incumbent = self.incumbent

    @classmethod
    def from_folder(cls,
                    folder: str,
                    ta_exec_dir: str,
                    file_format: str='SMAC3',
                    validation_format: str='NONE',
                    budget=None,
                    ):
        """Initialize scenario, runhistory and incumbent from folder, execute
        init-method of SMAC facade (so you could simply use SMAC-instances instead).

        Parameters
        ----------
        folder: string
            output-dir of this run -> this is also the 'id' for a single run in parallel optimization
        ta_exec_dir: string
            if the execution directory for the SMAC-run differs from the cwd, there might be problems loading instance-,
            feature- or PCS-files in the scenario-object. since instance- and PCS-files are necessary, specify the path
            to the execution-dir of SMAC here
        file_format: string
            from [SMAC2, SMAC3, CSV]
        validation_format: string
            from [SMAC2, SMAC3, CSV, NONE], in which format to look for validated data
        budget: int
            budget for this run-instance (only for budgeted optimization!)
        """
        self.logger = logging.getLogger("cave.ConfiguratorRun.{}".format(folder))
        self.logger.debug("Loading from \'%s\' with ta_exec_dir \'%s\' with file-format '%s' and validation-format %s. "
                          "Budget (if present): %s", folder, ta_exec_dir, file_format, validation_format, budget)

        self.validation_format = validation_format if validation_format != 'NONE' else None

        #### Read in data (scenario, runhistory & trajectory)
        reader = get_reader(file_format)

        scenario = self.reader.get_scenario()
        scenario_sanity_check(scenario, self.logger)
        original_runhistory = reader.get_runhistory(scenario.cs)
        validated_runhistory = None

        trajectory = reader.get_trajectory(cs=scenario.cs)

        if validation_format:
            reader = get_reader(validation_format)
            reader.scen = scenario
            validated_runhistory = reader.get_validated_runhistory(scenario.cs)
            self.logger.info("Found validated runhistory for \"%s\" and using it for evaluation. #configs in "
                             "validated rh: %d", folder, len(validated_runhistory.config_ids))

        self.__init__(
                 scenario,
                 original_runhistory,
                 validated_runhistory,
                 combined_runhistory,
                 epm_runhistory,
                 trajectory,
                 folder,
                 ta_exec_dir,
                 file_format,
                 validation_format,
                 budget=None,
                 )

    def get_incumbent(self):
        return self.solver.incumbent

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
            runs = rh.get_runs_for_config(c)
            evaluated = set([inst for inst, seed in runs])
            for i_name, i in [("train", self.train_inst),
                              ("test", self.test_inst)]:
                not_evaluated = set(i) - evaluated
                if len(not_evaluated) > 0:
                    self.logger.debug("RunHistory %s only evaluated on %d/%d %s-insts "
                                      "for %s in folder %s",
                                      name, len(i) - len(not_evaluated), len(i),
                                      i_name, c_name, self.folder)
                    return_value = False
        return return_value

    def _init_pimp_and_validator(self, alternative_output_dir=None):
        """Create ParameterImportance-object and use it's trained model for validation and further predictions We pass a
        combined (original + validated) runhistory, so that the returned model will be based on as much information as
        possible

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
                               parameters_to_evaluate=4,
                               save_folder=alternative_output_dir if alternative_output_dir else self.output_dir,
                               seed=self.rng.randint(1, 100000),
                               max_sample_size=self.pimp_max_samples,
                               fANOVA_pairwise=self.fanova_pairwise,
                               preprocess=False,
                               verbose=self.verbose_level != 'OFF',  # disable progressbars
                               )
        self.model = self.pimp.model

        # Validator (initialize without trajectory)
        self.validator = Validator(self.scenario, None, None)
        self.validator.epm = self.model

    @timing
    def _validate_default_and_incumbents(self, method, ta_exec_dir):
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
        self.logger.debug("Validating %s using %s!", self.folder, method)
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

