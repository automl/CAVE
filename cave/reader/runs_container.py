import logging
import tempfile
from typing import List

from numpy.random.mtrand import RandomState
from smac.runhistory.runhistory import RunHistory, DataOrigin

from cave.reader.configurator_run import ConfiguratorRun
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
        There will be `(n_budgets +1) * (m_parallel_execution + 1)` ConfiguratorRuns in CAVE, each representing the data
        of a specific budget-parallel-execution combination or an aggregated version..

        Aggregated entries can be accessed via a None-key.

        pr: parallel run, b: budget, agg: aggregated

        +----------+------+-----+------+-----------+
        |          | pr_1 | ... | pr_m | agg (None)|
        +==========================================+
        |b_1       |      |     |      |           +
        +----------+------+-----+------+-----------+
        |...       |      |     |      |           +
        +----------+------+-----+------+-----------+
        |b_2       |      |     |      |           +
        +----------+------+-----+------+-----------+
        |agg (None)|      |     |      |           +
        +----------+------+-----+------+-----------+

        The data is organized in folder2budgets as {pr : {b : path}} and in pRun2budget as {pr : {b : ConfiguratorRun}}.

        In the internal data-management there are three types of runhistories: *original*, *validated* and *epm*.

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
        ##########################################################################################
        #  Initialize and find suitable parameters                                               #
        ##########################################################################################
        self.logger = logging.getLogger(self.__module__ + '.' + self.__class__.__name__)

        self.folders = folders
        self.ta_exec_dirs = ta_exec_dirs if ta_exec_dirs else ['.' for f in folders]
        # Fix wrong input to ta_exec_dir
        if len(self.folders) < len(self.ta_exec_dirs):
            raise ValueError("Too many ta_exec_dirs ({}) compared to the number of folders ({})".format(
                                len(self.ta_exec_dirs), len(self.folders)))
        elif len(self.folders) > len(self.ta_exec_dirs):
            self.logger.info("Assuming ta_exec_dir is valid for all folders, expanding list")
            self.ta_exec_dirs.extend([self.ta_exec_dirs[0] for _ in range(len(self.folders) - len(self.ta_exec_dirs))])

        self.output_dir = output_dir if output_dir else tempfile.mkdtemp()

        if file_format.upper() == "AUTO":
            file_format = detect_fileformat(folders=self.folders)
        self.file_format = file_format
        self.validation_format = validation_format
        self.use_budgets = self.file_format == "BOHB"

        self.analyzing_options = load_default_options(analyzing_options, file_format)

        # Main focus on this mapping pRun2budget2data:
        self.pRun2budget = {None : {}}  # mapping parallel runs to their budgets
        self.runs_list = []  # Just put all ConfiguratorRun-objects here

        ##########################################################################################
        #  Convert if necessary, determine what folders and what budgets                         #
        ##########################################################################################
        # Both budgets and folders have "None" in the key-list for the aggregation over all available budgets/folders
        self.budgets = [None]
        if self.file_format == 'BOHB':
            self.logger.debug("Converting %d BOHB folders to SMAC-format", len(folders))
            hpbandster2smac = HpBandSter2SMAC()
            # Convert m BOHB-folders to m + n SMAC-folders
            # TODO make compatible with hpbandster
            self.folder2result, self.folder2budgets = hpbandster2smac.convert(self.folders, self.output_dir)
            self.budgets.extend(list(self.folder2result.values())[0].HB_config['budgets'])
            #if "DEBUG" in self.verbose_level:
            #    for f in folders:
            #        debug_f = os.path.join(output_dir, 'debug', os.path.basename(f))
            #        shutil.rmtree(debug_f, ignore_errors=True)
            #        shutil.copytree(f, debug_f)
        else:
            self.folder2budgets = {f : {None : f} for f in self.folders}

        ##########################################################################################
        #  Read in folders, where folders are parallel runs and for each parallel-run/budget     #
        #  combination there is one ConfiguratorRun-object (they can be easily aggregated)       #
        ##########################################################################################
        self.logger.debug("Reading in folders: %s with ta_exec_dirs: %s", str(folders), str(self.ta_exec_dirs))
        for f, ta_exec_dir in zip(self.folders, self.ta_exec_dirs):  # Iterating over parallel runs
            self.logger.debug("--Processing folder \"{}\" (and ta_exec_dir \"{}\")".format(f, ta_exec_dir))
            self.pRun2budget[f] = {}
            for b, path in self.folder2budgets[f].items():
                self.logger.debug("----Processing budget \"{}\" (and path: \"{}\")".format(b, path))
                # Using folder of (converted) data here
                try:
                    cr = ConfiguratorRun.from_folder(path,
                                                     ta_exec_dir,
                                                     self.analyzing_options,
                                                     file_format=self.file_format,
                                                     validation_format=self.validation_format,
                                                     budget=b,
                                                     output_dir = self.output_dir)
                except Exception as err:
                    self.logger.warning("Folder %s could with ta_exec_dir %s not be loaded, failed with error message: %s",
                                        f, ta_exec_dir, err)
                    self.logger.exception(err)
                    continue
                self.pRun2budget[f][b] = cr
                self.runs_list.append(cr)

        self.folders.append(None)
        self.logger.debug("folder2budgets: " + str(self.folder2budgets))
        self.logger.debug("pRun2budget: " + str(self.pRun2budget))

        self.scenario = self.runs_list[0].scenario

        if not self.get_all_runs():
            raise ValueError("None of the specified folders could be loaded.")

    def __getitem__(self, key):
        """ Return highest budget for given folder. """
        if self.use_budgets:
            return self.pRun2budget[key][self.get_highest_budget()]
        else:
            return self.pRun2budget[key][None]

    def get_bohb_results(self):
        if self.file_format == "BOHB":
            return list(self.folder2result.values())
        else:
            return None

    def get_all_runs(self):
        return self.runs_list

    def get_rng(self):
        return RandomState(42)

    def get_highest_budget(self):
        return max(self.get_budgets()) if self.use_budgets else None

    def get_budgets(self):
        budgets = set()
        for f in self.get_folders():
            budgets.update(set(self.pRun2budget[f].keys()))
        budgets = sorted([b for b in budgets if b is not None])
        self.logger.debug("Budgets: " + str(budgets))
        return budgets


    def get_runs_for_budget(self, target_b):
        runs = [[cr for b, cr in self.pRun2budget[f].items() if b == target_b] for f in self.pRun2budget.keys() if f is
                not None]
        # Flatten list
        runs = [a for b in runs for a in b]
        return runs

    def get_folders(self):
        return [folder for folder in self.pRun2budget.keys() if folder is not None]

    def get_runs_for_folder(self, f):
        return list(self.pRun2budget[f].values())

    def get_aggregated(self, keep_budgets=True, keep_folders=False):
        """ Collapse data-structure along a given "axis".

        Returns
        -------
        aggregated_runs: List[ConfiguratorRun]
            run(s) with aggregated data
        """
        if not self.use_budgets:
            keep_budgets = False

        if (not keep_budgets) and (not keep_folders):
            if None not in self.pRun2budget[None].keys():
                self.logger.debug("Aggregating all runs")
                aggregated = self._aggregate(self.get_all_runs())
                self.pRun2budget[None][None] = aggregated
            res = [self.pRun2budget[None][None]]
        elif keep_budgets and not keep_folders:
            for b in self.get_budgets():
                if b not in self.pRun2budget[None].keys():
                    self.logger.debug("Aggregating over parallel runs, keeping budgets")
                    self.pRun2budget[None][b] = self._aggregate(self.get_runs_for_budget(b))
            res = [self.pRun2budget[None][b] for b in self.get_budgets()]
        elif keep_folders and not keep_budgets:
            for f in self.get_folders():
                if None not in self.pRun2budget[f].keys():
                    self.logger.debug("Aggregating over budgets, keeping parallel runs")
                    self.pRun2budget[f][None] = self._aggregate(self.get_runs_for_folder(f))
            res = [self.pRun2budget[f][None] for f in self.get_folders()]
        else:
            res = self.runs_list
        self.logger.debug("Aggregated: {}".format(str([r.get_identifier() for r in res])))
        return res

    def _aggregate(self, runs):
        """
        """
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

        path_to_folder = runs[0].path_to_folder if len(set([r.path_to_folder for r in runs])) == 1 else None
        budget = runs[0].budget if len(set([r.budget for r in runs])) == 1 else None

        new_cr = ConfiguratorRun(runs[0].scenario,
                                 orig_rh,
                                 vali_rh,
                                 traj,
                                 self.analyzing_options,
                                 output_dir=self.output_dir,
                                 path_to_folder=path_to_folder,
                                 budget=budget,
                                 )
        return new_cr

