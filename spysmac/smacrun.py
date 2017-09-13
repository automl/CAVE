import os
import logging
from contextlib import contextmanager

from smac.optimizer.objective import average_cost
from smac.utils.io.input_reader import InputReader
from smac.runhistory.runhistory import RunKey, RunValue, RunHistory
from smac.scenario.scenario import Scenario
from smac.utils.io.traj_logging import TrajLogger
from smac.utils.validate import Validator

@contextmanager
def changedir(newdir):
    olddir = os.getcwd()
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(olddir)

class SMACrun(object):
    """
    SMACrun keeps all information on a specific SMAC run.
    """
    def __init__(self, folder, ta_exec_dir="."):
        """
        Parameters
        ----------
        folder: string
            output-dir of this run
        global_rh: RunHistory
            runhistory with runs from other (validated) SMAC-folders, so no
            runs need to be repeated multiple times (for example defaults).
        ta_exec_dir: string
            if the execution directory for the SMAC-run differs from the cwd,
            there might be problems loading instance-, feature- or PCS-files
            in the scenario-object. since instance- and PCS-files are necessary,
            specify the path to the execution-dir of SMAC here
        """
        self.logger = logging.getLogger("spysmac.SMACrun.{}".format(folder))
        in_reader = InputReader()

        self.folder = folder
        self.logger.debug("Loading from %s", folder)

        self.scen_fn = os.path.join(folder, 'scenario.txt')
        self.rh_fn = os.path.join(folder, 'runhistory.json')
        self.traj_fn = os.path.join(folder, "traj_aclib2.json")

        # Create Scenario (disable output_dir to avoid cluttering)
        self.scen = in_reader.read_scenario_file(self.scen_fn)
        self.scen['output_dir'] = ""
        with changedir(ta_exec_dir):
            self.scen = Scenario(self.scen)

        # Load runhistory and trajectory
        self.rh = RunHistory(average_cost)
        self.traj = TrajLogger.read_traj_aclib_format(fn=self.traj_fn,
                                                      cs=self.scen.cs)

        self.incumbent = self.traj[-1]['incumbent']
        self.train_inst = self.scen.train_insts
        self.test_inst = self.scen.test_insts

    def get_incumbent(self):
        """Return tuple (incumbent, loss)."""
        return (self.incumbent, self.rh.get_cost(self.incumbent))

