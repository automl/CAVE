import logging
import tempfile
from collections import OrderedDict
from typing import List

from numpy.random.mtrand import RandomState
from smac.runhistory.runhistory import RunHistory, DataOrigin

from cave.reader.configurator_run import ConfiguratorRun
from cave.reader.conversion.csv2smac import CSV2SMAC
from cave.reader.conversion.hpbandster2smac import HpBandSter2SMAC
from cave.utils.helpers import combine_trajectories, load_default_options, detect_fileformat


class RunsContainer(object):

    def __init__(self,
                 folders,
                 ta_exec_dirs=None,
                 output_dir=None,
                 file_format=None,
                 validation_format=None,
                 analyzing_options=None,
                 ):
        """
        Reads in optimizer runs. Converts data if necessary.

        SMAC3's RunHistory supports budgets from 0.12.0, so this container will by default keep one ConfiguratorRun per
        folder (assuming folders are parallel runs). Budgets are integrated in RunHistories per conversion.
        The RunHistory object provides an easy way to aggregate over parallel runs or budgets.

        The data is organized in self.data as {folder_name : ConfiguratorRun}.
        Aggregated or reduced ConfiguratorRuns are cached by their identifier (needs to be unique from context!)
          in self.cache as {identifier : ConfiguratorRun},-

        In the internal data-management there are three types of runhistories: *original*, *validated* and *epm*.
        They are saved in and provided by the ConfiguratorRuns

        * *original_rh* contain only runs that have been gathered during the optimization-process.
        * *validated_rh* may contain original runs, but also data that was not gathered iteratively during the
          optimization, but systematically through external validation of interesting configurations.
          Important: NO ESTIMATED RUNS IN `validated` RUNHISTORIES!
        * *epm_rh* contain runs that are gathered through empirical performance models.

        Runhistories are organized as follows:

        * each ConfiguratorRun has an *original_runhistory*- and a *combined_runhistory*-attribute
        * if available, each ConfiguratorRun's *validated_runhistory* contains
          a runhistory with validation-data gathered after the optimization
        * *combined_runhistory* always contains as many real runs as possible


        Parameters
        ----------
        folders: List[str]
            list of folders to read in
        ta_exec_dirs: List[str]
            optional, list of execution directories for target-algorithms (to find filepaths, etc.). If you're not sure,
            just set to current working directory (which it is by default).
        output_dir: str
            directory for output (temporary directory if not set)
        file_format: str
            from [SMAC2, SMAC3, BOHB, CSV] defines what file-format the optimizer result is in.
        validation_format: str
            from [SMAC2, SMAC3, BOHB, CSV] defines what file-format validation data is in.
        """
        ################################################################################################################
        #  Initialize and find suitable parameters                                                                     #
        ################################################################################################################
        self.logger = logging.getLogger(self.__module__ + '.' + self.__class__.__name__)

        self.folders = folders
        ta_exec_dirs = ta_exec_dirs if ta_exec_dirs else ['.']
        self.ta_exec_dirs = [ta_exec_dirs[0] for _ in range(len(folders))] if len(ta_exec_dirs) == 1 else ta_exec_dirs
        # Fix wrong input to ta_exec_dir
        if len(self.folders) < len(self.ta_exec_dirs):
            raise ValueError("ta_exec_dirs (# {}) compared to the number of folders ({})".format(
                                len(self.ta_exec_dirs), len(self.folders)))

        self.output_dir = output_dir if output_dir else tempfile.mkdtemp()

        if file_format.upper() == "AUTO":
            file_format = detect_fileformat(folders=self.folders)
            self.logger.info("Format of input detected automatically: %s", file_format)
        self.file_format = file_format
        self.validation_format = validation_format

        self.analyzing_options = load_default_options(analyzing_options, file_format)

        # Main focus on this mapping pRun2budget2data:
        self.data = OrderedDict()   # mapping parallel runs to their budgets
        self.cache = OrderedDict()  # Reuse already generated ConfiguratorRuns

        ################################################################################################################
        #  Convert if necessary, determine what folders and what budgets                                               #
        ################################################################################################################
        # Both budgets and folders have "None" in the key-list for the aggregation over all available budgets/folders
        input_data = {f : {} for f in self.folders}
        if self.file_format == 'BOHB':
            self.logger.debug("Converting %d BOHB folders to SMAC-format", len(folders))
            hpbandster2smac = HpBandSter2SMAC()
            result = hpbandster2smac.convert(self.folders, self.output_dir)
            input_data = result
        elif self.file_format == 'CSV':
            self.logger.debug("Check whether CSV-data needs to be split up (only if budgets are used)")
            csv2smac = CSV2SMAC()
            result = csv2smac.convert(self.folders, self.ta_exec_dirs, self.output_dir)
            input_data = result
            self.ta_exec_dirs = ['.' for _ in range(len(self.folders))]

        ################################################################################################################
        #  Read in folders, where folders are parallel runs and for each parallel-run                                  #
        #  there is one ConfiguratorRun-object (they can be easily aggregated)                                         #
        ################################################################################################################
        self.logger.debug("Reading in folders: %s with ta_exec_dirs: %s", str(self.folders), str(self.ta_exec_dirs))
        for f, ta_exec_dir in zip(self.folders, self.ta_exec_dirs):  # Iterating over parallel runs
            self.logger.debug("--Processing folder \"{}\" (and ta_exec_dir \"{}\")".format(f, ta_exec_dir))

            if all([x in input_data[f] for x in ['new_path', 'config_space', 'runhistory', 'scenario', 'trajectory']]):
                # Data has been converted and should therefore be available here
                self.logger.debug('Input data already read in for folder %s', f)
                self.logger.debug(list(input_data[f]['runhistory'].data.items())[:10])
                cr = ConfiguratorRun(
                    scenario=input_data[f].pop('scenario'),
                    original_runhistory=input_data[f].pop('runhistory'),
                    validated_runhistory=input_data[f].pop('validated_runhistory', None),
                    trajectory=input_data[f].pop('trajectory'),
                    options=self.analyzing_options,
                    path_to_folder=input_data[f].pop('new_path'),
                    ta_exec_dir=ta_exec_dir,
                    file_format=file_format,
                    validation_format=validation_format,
                    output_dir=self.output_dir)
                # Any format-specific information
                for k, v in input_data[f].items():
                    cr.share_information[k] = v
            else:
                # Data is in good readable SMAC3-format
                cr = ConfiguratorRun.from_folder(f,
                                                 ta_exec_dir,
                                                 self.analyzing_options,
                                                 file_format=self.file_format,
                                                 validation_format=self.validation_format,
                                                 output_dir = self.output_dir)
            self.data[f] = cr
        self.scenario = list(self.data.values())[0].scenario


    def __getitem__(self, key):
        """ Return highest budget for given folder. """
        return self.data[key]

    def get_run(self, folder, budget):
        return self._reduce_cr_to_budget(self.data[folder], [budget])

    def get_all_runs(self):
        return list(self.data.values())

    def get_rng(self):
        return RandomState(42)

    def get_highest_budget(self):
        return max(self.get_budgets()) if self.get_budgets() else None

    def get_budgets(self):
        budgets = set()
        for cr in self.data.values():
            budgets.update(cr.get_budgets())
        budgets = sorted([b for b in budgets if b is not None])
        self.logger.debug("Budgets: " + str(budgets))
        return budgets if len(budgets) > 0 else None

    def get_runs_for_budget(self, target_b):
        runs = [self._reduce_cr_to_budget(cr, [target_b]) for cr in self.get_all_runs()]
        return runs

    def get_folders(self):
        self.logger.debug('Folders: %s', list(self.data.keys()))
        return list(self.data.keys())

    def get_runs_for_folder(self, f):
        return self.data[f]

    def get_aggregated(self, keep_budgets=True, keep_folders=False):
        """ Collapse data-structure along a given "axis".

        Returns
        -------
        aggregated_runs: List[ConfiguratorRun]
            run(s) with aggregated data
        """
        if self.get_budgets() is None:
            keep_budgets = False

        if (not keep_budgets) and (not keep_folders):
            self.logger.debug("Aggregating all runs")
            res = [self._aggregate(self.get_all_runs())]
        elif keep_budgets and not keep_folders:
            self.logger.debug("Aggregating over parallel runs, keeping budgets")
            all_runs = self.get_all_runs()
            res = [self._aggregate([self._reduce_cr_to_budget(cr, [b]) for cr in all_runs]) for b in self.get_budgets()]
            assert len(self.get_budgets()) == len(res)
        elif keep_folders and not keep_budgets:
            res = self.get_all_runs()
        else:
            res = self.get_all_runs()
        self.logger.debug("Aggregated: {}".format(str([r.get_identifier() for r in res])))
        return res

    def _aggregate(self, runs):
        # path_to_folder is the concatenation of all the paths of the individual runs
        path_to_folder = '-'.join(sorted(list(set([r.path_to_folder for r in runs]))))
        # budgets are the union of individual budgets. if they are not the same for all runs (no usecase atm),
        #   they get an additional entry of the hash over the string of the combination to avoid false-positives
        budgets = [r.reduced_to_budgets for r in runs]
        budget_hash = ['budgetmix-%d' % (hash(str(budgets)))] if len(set([frozenset(b) for b in budgets])) != 1 else []
        budgets = [a for b in [x for x in budgets if x is not None] for a in b] + budget_hash

        if ConfiguratorRun.identify(path_to_folder, budgets) in self.cache:
            return self.cache[ConfiguratorRun.identify(path_to_folder, budgets)]

        orig_rh, vali_rh = RunHistory(), RunHistory()
        for run in runs:
            orig_rh.update(run.original_runhistory, origin=DataOrigin.INTERNAL)
            vali_rh.update(run.original_runhistory, origin=DataOrigin.INTERNAL)
            if run.validated_runhistory:
                vali_rh.update(run.validated_runhistory, origin=DataOrigin.EXTERNAL_SAME_INSTANCES)

        for rh_name, rh in [("original", orig_rh),
                            ("validated", vali_rh),
                            ]:
            self.logger.debug('Combined number of %s RunHistory data points: %d '
                              '# Configurations: %d. # Configurator runs: %d',
                              rh_name, len(rh.data), len(rh.get_all_configs()), len(runs))

        traj = combine_trajectories([run.trajectory for run in runs], self.logger)

        new_cr = ConfiguratorRun(runs[0].scenario,
                                 orig_rh,
                                 vali_rh,
                                 traj,
                                 self.analyzing_options,
                                 output_dir=self.output_dir,
                                 path_to_folder=path_to_folder,
                                 reduced_to_budgets=budgets,
                                 )

        self._cache(new_cr)
        return new_cr

    def _reduce_cr_to_budget(self, cr, keep_budgets):
        """Creates a new ConfiguratorRun without all the target algorithm runs that are not in the list of budgets.
        Will affect original, validated and epm-RunHistories as well as Trajectory"""
        if ConfiguratorRun.identify(cr.path_to_folder, keep_budgets) in self.cache:
            return self.cache[ConfiguratorRun.identify(cr.path_to_folder, keep_budgets)]

        def reduce_runhistory(rh, keep_budgets):
            if not isinstance(rh, RunHistory):
                self.logger.debug("This is not a RunHistory: %s", rh)
                return rh
            new_rh = RunHistory()
            for rk, rv in rh.data.items():
                if rk.budget in keep_budgets or rh.ids_config[rk.config_id] in [cr.default]:
                    new_rh.add(config=rh.ids_config[rk.config_id],
                               cost=rv.cost,
                               time=rv.time,
                               status=rv.status,
                               instance_id=rk.instance_id,
                               seed=rk.seed,
                               budget=rk.budget,
                               additional_info=rv.additional_info,
                               origin=rh.external[rk])
            return new_rh

        orig_rh = reduce_runhistory(cr.original_runhistory, keep_budgets)
        vali_rh = reduce_runhistory(cr.validated_runhistory, keep_budgets)
        trajectory = [entry for entry in cr.trajectory if (entry['incumbent'] in orig_rh.config_ids.keys())]

        if any([len(x) == 0 for x in [orig_rh.data, trajectory]]):
            self.logger.debug("Runhistory: %s, Trajectory: %s", str(orig_rh.data), str(trajectory))
            raise ValueError("Reducing to budget {} for ConfiguratorRun {} failed for runhistory or trajectory. Are "
                             "same budgets used for all parallel runs?".format(str(keep_budgets), cr.path_to_folder))

        new_cr = ConfiguratorRun(scenario=cr.scenario,
                                 original_runhistory=orig_rh,
                                 validated_runhistory=vali_rh,
                                 trajectory=trajectory,
                                 options=self.analyzing_options,
                                 output_dir=self.output_dir,
                                 path_to_folder=cr.path_to_folder,
                                 reduced_to_budgets=keep_budgets,
                                 )

        self.logger.debug("Reduced CR %s to CR %s", cr.get_identifier(), new_cr.get_identifier())

        self._cache(cr)

        return new_cr

    def _cache(self, configurator_run):
        self.cache[configurator_run.get_identifier()] = configurator_run