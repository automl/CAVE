import sys
import os
import logging
from collections import OrderedDict
from contextlib import contextmanager
from importlib import reload
import typing
from typing import Union, List
import tempfile
import copy
from functools import wraps
import shutil
import inspect

import numpy as np
from pandas import DataFrame

from smac.optimizer.objective import average_cost
from smac.runhistory.runhistory import RunHistory, DataOrigin
from smac.utils.io.input_reader import InputReader

from cave.reader.conversion.hpbandster2smac import HpBandSter2SMAC
from cave.reader.configurator_run import ConfiguratorRun

class RunsContainer(object):

    def __init__(self, folders, ta_exec_dirs=None, output_dir=None, file_format=None, validation_format=None):
        """
        Reads in optimizer runs. Converts data if necessary.
        There will be `(n_budgets +1) * (m_parallel_execution + 1)` ConfiguratorRuns in CAVE, each representing the data
        of a specific budget-parallel-execution combination or an aggregated version..

        Aggregated entries can be accessed via a None-key.

        pr: parallel run, b: budget, agg: aggregated
                  | pr_1 | ... | pr_m | agg (None)
        ------------------------------------------
        b_1       |      |     |      |
        ...       |      |     |      |
        b_2       |      |     |      |
        agg (None)|      |     |      |

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
        self.output_dir = output_dir if output_dir else tempfile.mkdtemp()

        # TODO: detect file_format...
        if not file_format:
            raise ValueError("Automatic file-format detection will be implemented soon, until then please specify.")
        #if not validation_format:
        #    raise ValueError("Automatic file-format detection for validation data will be implemented soon, until then please specify.")
        self.file_format = file_format
        self.validation_format = validation_format
        self.use_budgets = self.file_format == "BOHB"

        # Main focus on this mapping run2budget2data:
        self.run2budget = {None : None}  # mapping parallel runs to their budgets
        self.runs_list = []  # Just put all ConfiguratorRun-objects here

        ##########################################################################################
        #  Convert if necessary, detemine what folders and what budgets                          #
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
        else:
            self.folder2budgets = {f : {None : f} for f in self.folders}
        self.folders.append(None)

        ##########################################################################################
        #  Read in folders, where folders are parallel runs and for each parallel-run/budget     #
        #  combination there is one ConfiguratorRun-object (they can be easily aggregated)       #
        ##########################################################################################
        self.logger.debug("Folders: %s, ta_exec_dirs: %s", str(folders), str(self.ta_exec_dirs))
        for f, ta_exec_dir in zip(self.folders, self.ta_exec_dirs):  # Iterating over parallel runs
            self.run2budget[f] = {}
            for b, path in self.folder2budgets[f].items():
                # Using folder of (converted) data here
                self.logger.debug(path)
                cr = ConfiguratorRun.from_folder(path,
                                                 ta_exec_dir,
                                                 self.file_format,
                                                 self.validation_format,
                                                 b,
                                                 self.output_dir)
                self.run2budget[f][b] = cr
                self.runs_list.append(cr)

        self.scenario = self.runs_list[0].scenario

        if not self.get_all_runs():
            raise ValueError("None of the specified folders could be loaded.")

    def __getitem__(self, key):
        """ Return highest budget for given folder. """
        if self.use_budgets:
            return self.run2budgets[key][self.get_highest_budget()]
        else:
            return self.run2budget[key][None]

    def get_all_runs(self):
        return self.runs_list

    def get_highest_budget(self):
        return max(self.get_budgets) if self.use_budgets else None

    def get_budgets(self):
        budgets = set()
        for f in self.get_folders():
            budgets.update(set(self.run2budgets[f].values()))
        return [b for b in budgets if b is not None]

    def get_runs_for_budget(self, b):
        return list(self.budgets2folders.values())

    def get_folders(self):
        return [folder for folder in self.run2budgets.keys() if folder is not None]

    def get_runs_for_folder(self, f):
        return list(self.folders2budgets[f].values())

    def get_aggregated(self, keep_budgets=True, keep_folders=False):
        """ Collapse data-structure along a given "axis".

        Returns
        -------
        aggregated_runs: either ConfiguratorRun or Dict(str->ConfiguratorRun)
            run(s) with aggregated data
        """
        if not self.use_budgets:
            keep_budgets = False

        # TODO foldername of aggregated runs
        if (not keep_budgets) and (not keep_folders):
            if not None in self.run2budget[None].keys():
                self.logger.debug("Aggregating all runs")
                aggregated = aggregate_configurator_runs(runs)
                self.run2budget[None][None] = aggregated
            return self.run2budget[None][None]
        elif keep_budgets:
            for b in self.get_budgets():
                if not b in self.run2budget[None].keys():
                    self.logger.debug("Aggregating over parallel runs, keeping budgets")
                    self.run2budget[None][b] = self._aggregate(self.get_runs_for_budget(b))
            return {b : self.run2budget[None][b] for b in self.get_budgets()}
        elif keep_folders:
            for f in self.get_folders():
                if not None in self.run2budget[f].keys():
                    self.logger.debug("Aggregating over parallel runs, keeping budgets")
                    self.run2budget[f][None] = self._aggregate(self.get_runs_for_folder(f))
            return {f : self.run2budget[f][None] for f in self.get_folders()}
        else:
            return self.runs

    def _aggregate(self, runs):
        """
        """
        orig_rh, vali_rh = RunHistory(average_cost), RunHistory(average_cost)
        for run in runs:
            orig_rh.update(run.original_runhistory, origin=DataOrigin.INTERNAL)
            vali_rh.update(run.original_runhistory, origin=DataOrigin.INTERNAL)
            if run.validated_runhistory:
                vali_rh.update(run.validated_runhistory, origin=DataOrigin.EXTERNAL_SAME_INSTANCES)

        # Estimate missing costs for [def, inc1, inc2, ...]
        self._validate_default_and_incumbents(self.validation_method, self.ta_exec_dir)
        self.global_epm_rh.update(self.global_validated_rh)

        for rh_name, rh in [("original", self.global_original_rh),
                            ("validated", self.global_validated_rh),
                            ("epm", self.global_epm_rh)]:
            self.logger.debug('Combined number of %s RunHistory data points: %d '
                              '# Configurations: %d. # Configurator runs: %d',
                              rh_name, len(rh.data), len(rh.get_all_configs()), len(self.runs))

        # Sort runs (best first)
        runs = sorted(runs, key=lambda run: self.global_epm_rh.get_cost(run.solver.incumbent))
        self.best_run = runs[0]

        self.incumbent = self.pimp.incumbent = self.best_run.solver.incumbent
        self.logger.debug("Overall best run: %s, with incumbent: %s", self.best_run.folder, self.incumbent)

        return ConfiguratorRun(scenario, orig_rh, vali_rh, traj, folder, '.',
                               file_format, validation_format, budget)
