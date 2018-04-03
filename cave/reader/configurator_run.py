import os
import logging
import shutil
from contextlib import contextmanager
from typing import Union

from smac.facade.smac_facade import SMAC
from smac.optimizer.objective import average_cost
from smac.utils.io.input_reader import InputReader
from smac.runhistory.runhistory import RunKey, RunValue, RunHistory, DataOrigin
from smac.scenario.scenario import Scenario
from smac.utils.io.traj_logging import TrajLogger
from smac.utils.validate import Validator
from smac.optimizer.objective import _cost

from cave.reader.smac3_reader import SMAC3Reader
from cave.reader.smac2_reader import SMAC2Reader


class ConfiguratorRun(SMAC):
    """
    ConfiguratorRuns load and maintain information about individual configurator
    runs. There are three supported formats: SMAC3, SMAC2 and CSV
    This class is responsible for providing a scenario, a runhistory and a
    trajectory.
    """
    def __init__(self, folder: str, ta_exec_dir: str='.', file_format: str='SMAC3'):
        """Initialize scenario, runhistory and incumbent from folder, execute
        init-method of SMAC facade (so you could simply use SMAC-instances instead)

        Parameters
        ----------
        folder: string
            output-dir of this run
        ta_exec_dir: string
            if the execution directory for the SMAC-run differs from the cwd,
            there might be problems loading instance-, feature- or PCS-files
            in the scenario-object. since instance- and PCS-files are necessary,
            specify the path to the execution-dir of SMAC here
        file_format: string
            from [SMAC2, SMAC3, CSV]
        """
        self.logger = logging.getLogger("cave.SMACrun.{}".format(folder))
        self.folder = folder
        self.ta_exec_dir = ta_exec_dir
        self.logger.debug("Loading from %s", folder)
        self.logger.debug("ta_exec_dir %s", ta_exec_dir)

        if file_format == 'SMAC3':
            self.reader = SMAC3Reader(folder, ta_exec_dir)
        elif file_format == 'SMAC2':
            self.reader = SMAC2Reader(folder, ta_exec_dir)
        elif file_format == 'CSV':
            self.reader = CSVReader(folder, ta_exec_dir)
        else:
            raise ValueError("%s not supported as file-format" % file_format)

        self.scen = self.reader.get_scenario()
        self.original_runhistory, self.validated_runhistory = self.reader.get_runhistory(self.scen.cs)
        self.traj = self.reader.get_trajectory(cs=self.scen.cs)

        # Check validated runhistory for completeness
        if (self.validated_runhistory and
            self._check_rh_for_inc_and_def(self.validated_runhistory)):
            self.logger.info("Found validated runhistory for \"%s\" and using "
                              "it for evaluation. #configs in validated rh: %d",
                              self.folder, len(self.validated_runhistory.config_ids))
        elif self.validated_runhistory:
            self.logger.warning("Found validated runhistory, but it's not "
                                "evaluated for default and incumbent, so "
                                "it's disregarded.")
            self.validated_runhistory = False

        self.combined_runhistory = RunHistory(average_cost)
        self.combined_runhistory.update(self.original_runhistory,
                                        origin=DataOrigin.INTERNAL)
        if self.validated_runhistory:
            self.combined_runhistory.update(self.validated_runhistory,
                                            origin=DataOrigin.EXTERNAL_SAME_INSTANCES)


        self.default = self.scen.cs.get_default_configuration()
        self.incumbent = self.traj[-1]['incumbent']
        self.train_inst = self.scen.train_insts
        self.test_inst = self.scen.test_insts

        # Initialize SMAC-object
        super().__init__(scenario=self.scen, runhistory=self.combined_runhistory)
                #restore_incumbent=incumbent)
        # TODO use restore, delete next line
        self.solver.incumbent = self.incumbent

    def get_incumbent(self):
        return self.solver.incumbent

    def _check_rh_for_inc_and_def(self, rh):
        """
        Check if default and incumbent are evaluated on all instances in this rh

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
                    self.logger.warning("RunHistory only evaluated on %d/%d %s-insts "
                                        "for %s in folder %s",
                                        len(i) - len(not_evaluated), len(i),
                                        i_name, c_name, self.folder)
                    return_value = False
        return return_value

    def _read_files(self, file_format):
        """Runhistories should be in target directory. Allowed names are:
            runhistory.json, runhistory.csv,
            validated_runhistory.json, validated_runhistory.csv

        Returns
        -------
        (path, rh)
        """
        if file_format == 'smac3':
            return (rh_fn, rh)
        elif file_format == 'smac2':
            rh = SMAC2Reader().read_from_csv(self.folder,
                                             scenario=self.scenario)

        elif file_format == 'csv':
            configurations = os.path.join(self.folder, "configurations.csv")

            rh = CSV2RH().read_from_csv(rh_fn,
                                        pcs=self.scenario.cs,
                                        configurations=configurations,
                                        train_inst=self.scenario.train_insts,
                                        test_inst=self.scenario.test_insts,
                                        instance_features=self.scenario.feature_dict,
                                        )
        return (rh_fn, rh)

