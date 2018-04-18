import re
import os
import shutil
import csv
import numpy as np
import pandas as pd

from ConfigSpace import Configuration, c_util
from ConfigSpace.hyperparameters import IntegerHyperparameter, FloatHyperparameter
from smac.optimizer.objective import average_cost
from smac.utils.io.input_reader import InputReader
from smac.runhistory.runhistory import RunKey, RunValue, RunHistory, DataOrigin
from smac.utils.io.traj_logging import TrajLogger
from smac.scenario.scenario import Scenario

from cave.reader.base_reader import BaseReader, changedir
from cave.reader.csv2rh import CSV2RH
from cave.utils.io import load_csv_to_pandaframe, load_config_csv

class CSVReader(BaseReader):

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
        self.scen = scen
        return scen

    def get_runhistory(self, cs):
        """Reads runhistory in csv-format:

        +--------------------+--------------------+------+------+------+--------+
        |      config_id     |  instance_id       | cost | time | seed | status |
        +====================+====================+======+======+======+========+
        | name of config 1   | name of instance 1 | ...  |  ... | ...  |  ...   |
        +--------------------+--------------------+------+------+------+--------+
        |         ...        |          ...       | ...  |  ... | ...  |  ...   |
        +--------------------+--------------------+------+------+------+--------+

        Returns:
        --------
        (rh, validated_rh): RunHistory, Union[False, RunHistory]
            runhistory and (if available) validated runhistory
        """

        validated_rh = False
        rh_fn = os.path.join(self.folder, 'runhistory.csv')
        if not os.path.exists(rh_fn):
            raise FileNotFoundError("Specified format is \'CSV\', but no "
                                    "\'runhistory.csv\'-file could be found "
                                    "in %s" % self.folder)
        self.logger.debug("Runhistory loaded as csv from %s", rh_fn)
        configs_fn = os.path.join(self.folder, 'configurations.csv')
        if os.path.exists(configs_fn):
            self.logger.debug("Found \'configurations.csv\' in %s." % self.folder)
            self.configurations = load_config_csv(configs_fn, self.scen.cs, self.logger)[1]
        else:
            raise ValueError("No \'configurations.csv\' in %s." % self.folder)

        rh = CSV2RH().read_csv_to_rh(rh_fn,
                                     pcs=self.scen.cs,
                                     configurations=self.configurations,
                                     train_inst=self.scen.train_insts,
                                     test_inst=self.scen.test_insts,
                                     instance_features=self.scen.feature_dict,
                                     )

        return (rh, validated_rh)

    def get_trajectory(self, cs):
        """Reads `self.folder/trajectory.csv`, expected format:

        +----------+------+----------------+-----------+
        | cpu_time | cost | wallclock_time | incumbent |
        +==========+======+================+===========+
        | ...      | ...  | ...            | ...       |
        +----------+------+----------------+-----------+
        """
        traj_fn = os.path.join(self.folder, 'trajectory.csv')
        if not os.path.exists(traj_fn):
            self.logger.warning("Specified format is \'CSV\', but no "
                                "\'../trajectory\'-file could be found "
                                "at %s" % self.traj_fn)

        csv_data = load_csv_to_pandaframe(traj_fn, self.logger)
        traj = []
        def add_to_traj(row):
            new_entry = {}
            new_entry['cpu_time'] = row['cpu_time']
            new_entry['total_cpu_time'] = None
            new_entry["wallclock_time"] = row['wallclock_time']
            new_entry["evaluations"] = -1
            new_entry["cost"] = row["cost"]
            new_entry["incumbent"] = self.configurations[row["incumbent"]]
            traj.append(new_entry)
        csv_data.apply(add_to_traj, axis=1)
        return traj
