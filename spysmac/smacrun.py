import os
import logging as log
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
        self.logger = log.getLogger("spysmac.SMACrun.{}".format(folder))
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

    def validate(self, ta_exec_dir, global_rh):
        """Validate this run

        Parameters
        ----------
        ta_exec_dir: string
            directory from which to execute target algorithm

        Returns
        -------
        self.rh: RunHistory
            validated runhistory
        """
        # Update rh with global rh to avoid rerunning target algorithm runs
        self.rh.load_json(self.rh_fn, self.scen.cs)
        self.rh.update(global_rh)

        # Generate missing data via validation
        self.logger.info("Validating to complete data, saving validated "
                         "runhistory in %s.")
        with changedir(ta_exec_dir):
            validator = Validator(self.scen, self.traj, "")
            self.rh = validator.validate('def+inc', 'train+test', 1, -1)
        return self.rh

    def get_incumbent(self):
        """Return tuple (incumbent, loss)."""
        return (self.incumbent, self.rh.get_cost(self.incumbent))

    def get_loss_per_instance(self, conf, aggregate=None):
        """
        Aggregates loss for configuration on evaluated instances over seeds.

        Parameters:
        -----------
        conf: Configuration
            configuration to evaluate
        aggregate: function or None
            used to aggregate loss over different seeds, takes a list as
            argument

        Returns:
        --------
        loss: dict(instance->loss)
            loss per instance (aggregated or as list per seed)
        """
        # Check if config is in runhistory
        conf_id = self.rh.config_ids[conf]

        # Map instances to seeds in dict
        runs = self.rh.get_runs_for_config(conf)
        instance_seeds = dict()
        for r in runs:
            i, s = r
            if i in instance_seeds:
                instance_seeds[i].append(s)
            else:
                instance_seeds[i] = [s]

        # Get loss per instance
        instance_losses = {i: [self.rh.data[RunKey(conf_id, i, s)].cost for s in
                              instance_seeds[i]] for i in instance_seeds}

        # Aggregate:
        if aggregate:
            instance_losses = {i: aggregate(instance_losses[i]) for i in instance_losses}

        return instance_losses
