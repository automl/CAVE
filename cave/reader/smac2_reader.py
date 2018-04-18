import re
import os
import shutil
import csv
import numpy as np
import pandas as pd

from ConfigSpace import Configuration, c_util
from ConfigSpace.hyperparameters import IntegerHyperparameter, FloatHyperparameter
from ConfigSpace.util import deactivate_inactive_hyperparameters, fix_types
from smac.optimizer.objective import average_cost
from smac.utils.io.input_reader import InputReader
from smac.runhistory.runhistory import RunKey, RunValue, RunHistory, DataOrigin
from smac.utils.io.traj_logging import TrajLogger
from smac.scenario.scenario import Scenario

from cave.reader.base_reader import BaseReader, changedir
from cave.reader.csv2rh import CSV2RH

class SMAC2Reader(BaseReader):

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
        Expects the following files:

        - `self.folder/runs_and_results(...).csv`
        - `self.folder/paramstrings(...).csv`

        Returns
        -------
        (rh, validated_rh): RunHistory, Union[False, RunHistory]
            runhistory and (if available) validated runhistory
        """

        validated_rh = False
        rh_fn = re.search(r'runs\_and\_results.*?\.csv', str(os.listdir(self.folder)))
        if not rh_fn:
            raise FileNotFoundError("Specified format is \'SMAC2\', but no "
                                    "\'runs_and_results\'-file could be found "
                                    "in %s" % self.folder)
        rh_fn = os.path.join(self.folder, rh_fn.group())
        self.logger.debug("Runhistory loaded as csv from %s", rh_fn)
        configs_fn = re.search(r'paramstrings.*?\.txt', str(os.listdir(self.folder)))
        if not configs_fn:
            raise FileNotFoundError("Specified format is \'SMAC2\', but no "
                                    "\'paramstrings\'-file could be found "
                                    "in %s" % self.folder)
        configs_fn = os.path.join(self.folder, configs_fn.group())
        self.logger.debug("Configurations loaded from %s", configs_fn)
        # Translate smac2 to csv
        with open(rh_fn, 'r') as csv_file:
            csv_data = list(csv.reader(csv_file, delimiter=',',
                                       skipinitialspace=True))
        header, csv_data = csv_data[0], np.array([csv_data[1:]])[0]
        csv_data = pd.DataFrame(csv_data, columns=header)
        csv_data = csv_data.apply(pd.to_numeric, errors='ignore')
        self.logger.debug("Headers: " + str(list(csv_data.columns.values)))
        data = pd.DataFrame()
        data["config_id"] = csv_data["Run History Configuration ID"]
        data["instance_id"] = csv_data["Instance ID"].apply(lambda x:
                self.scen.train_insts[x-1])
        data["seed"] = csv_data["Seed"]
        data["time"] = csv_data["Runtime"]
        if self.scen.run_obj == 'runtime':
            data["cost"] = csv_data["Runtime"]
        else:
            data["cost"] = csv_data["Run Quality"]
        data["status"] = csv_data["Run Result"]

        # Load configurations
        with open(configs_fn, 'r') as csv_file:
            csv_data = list(csv.reader(csv_file, delimiter=',',
                                       skipinitialspace=True))
        configurations = {}  # id to config
        for row in csv_data:
            config_id = int(re.match(r'^(\d*):', row[0]).group(1))
            params = [re.match(r'^\d*: (.*)', row[0]).group(1)]
            params.extend(row[1:])
            #self.logger.debug(params)
            matches = [re.match(r'(.*)=\'(.*)\'', p) for p in params]
            values = {m.group(1) : m.group(2) for m in matches}
            rs = np.random.RandomState()
            for name, value in values.items():
                if isinstance(cs.get_hyperparameter(name), IntegerHyperparameter):
                    values[name] = int(value)
                if isinstance(cs.get_hyperparameter(name), FloatHyperparameter):
                    values[name] = float(value)
            configurations[config_id] = self._remove_inactive(cs, Configuration(cs,
                                                    values=values,
                                                    allow_inactive_with_values=True),
                                                    values)
        self.configurations = configurations

        names, feats = self.scen.feature_names, self.scen.feature_dict
        rh = CSV2RH().read_csv_to_rh(data,
                                     pcs=cs,
                                     configurations=configurations,
                                     train_inst=self.scen.train_insts,
                                     test_inst=self.scen.test_insts,
                                     instance_features=feats)

        return (rh, validated_rh)

    def get_trajectory(self, cs):
        """Expects the following files:

        - `self.folder/traj-run-(...).csv`
        """
        traj_fn = re.search(r'traj-run-\d*.txt', str(os.listdir(os.path.join(self.folder, '..'))))
        if not traj_fn:
            raise FileNotFoundError("Specified format is \'SMAC2\', but no "
                                    "\'../traj-run\'-file could be found "
                                    "in %s" % self.folder)
        traj_fn = os.path.join(self.folder, '..', traj_fn.group())
        with open(traj_fn, 'r') as csv_file:
            csv_data = list(csv.reader(csv_file, delimiter=',',
                                       skipinitialspace=True))
        header, csv_data = csv_data[0][:-1], np.array([csv_data[1:]])[0]
        csv_data = pd.DataFrame(np.delete(csv_data, np.s_[5:], axis=1), columns=header)
        csv_data = csv_data.apply(pd.to_numeric, errors='ignore')
        traj = []
        def add_to_traj(row):
            new_entry = {}
            new_entry['cpu_time'] = row['CPU Time Used']
            new_entry['total_cpu_time'] = None
            new_entry["wallclock_time"] = row['Wallclock Time']
            new_entry["evaluations"] = -1
            new_entry["cost"] = row["Estimated Training Performance"]
            new_entry["incumbent"] = self.configurations[row["Incumbent ID"]]
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
