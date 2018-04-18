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
        """
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
            print(self.configurations)
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

    def _remove_inactive(self, cs, configuration, values):
        num_hyperparameters = len(cs._hyperparameters)

        unconditional_hyperparameters = cs.get_all_unconditional_hyperparameters()
        hyperparameters_with_children = list()

        _forbidden_clauses_unconditionals = []
        _forbidden_clauses_conditionals = []
        for clause in cs.get_forbiddens():
            based_on_conditionals = False
            for subclause in clause.get_descendant_literal_clauses():
                if subclause.hyperparameter.name not in unconditional_hyperparameters:
                    based_on_conditionals = True
                    break
            if based_on_conditionals:
                _forbidden_clauses_conditionals.append(clause)
            else:
                _forbidden_clauses_unconditionals.append(clause)

        for uhp in unconditional_hyperparameters:
            children = cs._children_of[uhp]
            if len(children) > 0:
                hyperparameters_with_children.append(uhp)
        vector = configuration.get_array()
        return Configuration(
                      cs,
                      vector=c_util.correct_sampled_array(
                          vector,
                          _forbidden_clauses_unconditionals,
                          _forbidden_clauses_conditionals,
                          hyperparameters_with_children,
                          len(cs._hyperparameters),
                          cs.get_all_unconditional_hyperparameters(),
                          cs._hyperparameter_idx,
                          cs._parent_conditions_of,
                          cs._parents_of,
                          cs._children_of,
                      ))

        new_config = {}
        vector = configuration.get_array()
        for hp_name, hyperparameter in cs._hyperparameters.items():
            hp_value = vector[cs._hyperparameter_idx[hp_name]]
            active = True
            conditions = cs._parent_conditions_of[hyperparameter.name]
            for condition in conditions:

                parent_vector_idx = condition.get_parents_vector()

                # if one of the parents is None, the hyperparameter cannot be
                # active! Else we have to check this
                # Note from trying to optimize this - this is faster than using
                # dedicated numpy functions and indexing
                if any([vector[i] != vector[i] for i in parent_vector_idx]):
                    active = False
                    break

                else:
                    if not condition.evaluate_vector(vector):
                        active = False
                    break
            if active:
                new_config[hp_name] = values[hp_name]
        #return configuration
        return Configuration(cs, values=new_config)
