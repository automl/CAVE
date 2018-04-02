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
        (rh, validated_rh): RunHistory, Union[False, RunHistory]
            runhistory and (if available) validated runhistory
        """
        rh_fn = os.path.join(self.folder, 'runhistory.json')
        validated_rh_fn = os.path.join(self.folder, 'validated_runhistory.json')
        rh = RunHistory(average_cost)
        rh.load_json(rh_fn, cs)
        try:
            validated_rh = RunHistory(average_cost)
            validated_rh.load_json(validated_rh_fn, cs)
        except FileNotFoundError:
            self.logger.debug("No validated runhistory for \"%s\" found "
                              "(probably ok)" % self.folder)
            validated_rh = False

        return (rh, validated_rh)

    def get_trajectory(self, cs):
        traj_fn = os.path.join(self.folder, 'traj_aclib2.json')
        traj_old_fn = os.path.join(self.folder, 'traj_old.csv')
        traj = TrajLogger.read_traj_aclib_format(fn=traj_fn, cs=cs)
        return traj
