import os
import logging
import shutil
from contextlib import contextmanager
from typing import Union
import glob

from smac.facade.smac_facade import SMAC
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

class SMACrun(SMAC):
    """
    SMACrun keeps all information on a specific SMAC run. Extends the standard
    SMAC-facade.
    """
    def __init__(self, folder: str, ta_exec_dir: Union[str, None]=None):
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
        """
        run_1_existed = os.path.exists('run_1')
        self.logger = logging.getLogger("cave.SMACrun.{}".format(folder))
        in_reader = InputReader()

        self.folder = folder
        self.logger.debug("Loading from %s", folder)

        split_folder = os.path.split(folder)
        if split_folder[0] and ta_exec_dir is None:
            ta_exec_dir = split_folder[0]
        elif ta_exec_dir is None:
            ta_exec_dir = '.'
        else:
            ta_exec_dir = glob.glob(ta_exec_dir, recursive=True)
            candidates = []
            for f in ta_exec_dir:
                if f in split_folder[0] or split_folder[0] in f:
                    candidates.append(f)
            ta_exec_dir = list(sorted(candidates, key=lambda x: len(x), reverse=True))[0]

        self.scen_fn = os.path.join(folder, 'scenario.txt')
        self.rh_fn = os.path.join(folder, 'runhistory.json')
        self.traj_fn = os.path.join(folder, 'traj_aclib2.json')
        self.traj_old_fn = os.path.join(folder, 'traj_old.csv')

        # Create Scenario (disable output_dir to avoid cluttering)
        scen_dict = in_reader.read_scenario_file(self.scen_fn)
        scen_dict['output_dir'] = ""
        with changedir(ta_exec_dir):
            self.scen = Scenario(scen_dict)

        # Load runhistory and trajectory
        self.runhistory = RunHistory(average_cost)
        self.runhistory.update_from_json(self.rh_fn, self.scen.cs)
        self.traj = TrajLogger.read_traj_aclib_format(fn=self.traj_fn,
                                                      cs=self.scen.cs)

        incumbent = self.traj[-1]['incumbent']
        self.train_inst = self.scen.train_insts
        self.test_inst = self.scen.test_insts

        # Initialize SMAC-object
        super().__init__(scenario=self.scen, runhistory=self.runhistory)
                #restore_incumbent=incumbent)
        # TODO use restore, delete next line
        self.solver.incumbent = incumbent

        if (not run_1_existed) and os.path.exists('run_1'):
            shutil.rmtree('run_1')

    def get_incumbent(self):
        return self.solver.incumbent
