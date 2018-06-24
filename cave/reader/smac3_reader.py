import os
import shutil

from smac.optimizer.objective import average_cost
from smac.utils.io.input_reader import InputReader
from smac.runhistory.runhistory import RunKey, RunValue, RunHistory, DataOrigin
from smac.utils.io.traj_logging import TrajLogger
from smac.scenario.scenario import Scenario

from cave.reader.base_reader import BaseReader, changedir

class SMAC3Reader(BaseReader):

    def get_scenario(self):
        run_1_existed = os.path.exists('run_1')
        in_reader = InputReader()
        # Create Scenario (disable output_dir to avoid cluttering)
        scen_fn = os.path.join(self.folder, 'scenario.txt')
        scen_dict = in_reader.read_scenario_file(scen_fn)
        scen_dict['output_dir'] = ""
        with changedir(self.ta_exec_dir):
            self.logger.debug("Creating scenario from \"%s\"", self.ta_exec_dir)
            scen = Scenario(scen_dict)

        if (not run_1_existed) and os.path.exists('run_1'):
            shutil.rmtree('run_1')
        return scen

    def get_runhistory(self, cs):
        """
        Returns:
        --------
        rh: RunHistory
            runhistory
        """
        rh_fn = os.path.join(self.folder, 'runhistory.json')
        validated_rh_fn = os.path.join(self.folder, 'validated_runhistory.json')
        rh = RunHistory(average_cost)
        try:
            rh.load_json(rh_fn, cs)
        except FileNotFoundError:
            self.logger.warning("%s not found. trying to read SMAC3-output, "
                                "if that's not correct, change it with the "
                                "--format option!", rh_fn)
            raise
        return rh

    def get_validated_runhistory(self, cs):
        """
        Returns:
        --------
        validated_rh: RunHistory
            runhistory with validation-data, if available
        """
        rh_fn = os.path.join(self.folder, 'validated_runhistory.json')
        rh = RunHistory(average_cost)
        try:
            rh.load_json(rh_fn, cs)
        except FileNotFoundError:
            self.logger.warning("%s not found. trying to read SMAC3-validation-output, "
                                "if that's not correct, change it with the "
                                "--validation_format option!", rh_fn)
            raise
        rh

    def get_trajectory(self, cs):
        traj_fn = os.path.join(self.folder, 'traj_aclib2.json')
        traj = TrajLogger.read_traj_aclib_format(fn=traj_fn, cs=cs)
        return traj
