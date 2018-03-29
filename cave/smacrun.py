import os
import logging
import shutil
from contextlib import contextmanager
from typing import Union

from smac.facade.smac_facade import SMAC
from smac.optimizer.objective import average_cost
from smac.utils.io.input_reader import InputReader
from smac.runhistory.runhistory import RunKey, RunValue, RunHistory
from smac.scenario.scenario import Scenario
from smac.utils.io.traj_logging import TrajLogger
from smac.utils.validate import Validator
from smac.optimizer.objective import _cost

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
        self.logger.info(split_folder)
        if ta_exec_dir is None:
            ta_exec_dir = '.'

        self.scen_fn = os.path.join(folder, 'scenario.txt')
        self.rh_fn = os.path.join(folder, 'runhistory.json')
        self.validated_rh_fn = os.path.join(folder, 'validated_runhistory.json')
        self.traj_fn = os.path.join(folder, 'traj_aclib2.json')
        self.traj_old_fn = os.path.join(folder, 'traj_old.csv')

        # Create Scenario (disable output_dir to avoid cluttering)
        scen_dict = in_reader.read_scenario_file(self.scen_fn)
        scen_dict['output_dir'] = ""
        with changedir(ta_exec_dir):
            self.scen = Scenario(scen_dict)

        # Load trajectory
        self.traj = TrajLogger.read_traj_aclib_format(fn=self.traj_fn,
                                                      cs=self.scen.cs)
        self.default = self.scen.cs.get_default_configuration()
        self.incumbent = self.traj[-1]['incumbent']
        self.train_inst = self.scen.train_insts
        self.test_inst = self.scen.test_insts

        # Load runhistory
        self.original_runhistory = RunHistory(average_cost)
        self.original_runhistory.update_from_json(self.rh_fn, self.scen.cs)
        if os.path.exists(self.validated_rh_fn):
            self.validated_runhistory = RunHistory(average_cost)
            self.validated_runhistory.update_from_json(self.validated_rh_fn, self.scen.cs)
            self.logger.debug("Found validated runhistory for \"%s\" and using "
                              "it for evaluation. #configs in validated rh: %d",
                              self.folder, len(self.validated_runhistory.config_ids))
            # Check validated runhistory for completeness
            if not self._check_rh_for_inc_and_def(self.validated_runhistory):
                self.validated_runhistory = False
        else:
            self.logger.debug("No validated runhistory for \"%s\" found (probably ok)")
            self.validated_runhistory = False
        self.combined_runhistory = RunHistory(average_cost)
        self.combined_runhistory.update(self.original_runhistory)
        if self.validated_runhistory:
            self.combined_runhistory.update(self.validated_runhistory)


        # for config in self.traj:
        #     time = config['wallclock_time']
        #     config = config['incumbent']
        #     c = _cost(config, self.runhistory, self.runhistory.get_runs_for_config(config))
        #     print(len(c), time)

        # Initialize SMAC-object
        super().__init__(scenario=self.scen, runhistory=self.combined_runhistory)
                #restore_incumbent=incumbent)
        # TODO use restore, delete next line
        self.solver.incumbent = self.incumbent

        if (not run_1_existed) and os.path.exists('run_1'):
            shutil.rmtree('run_1')

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
