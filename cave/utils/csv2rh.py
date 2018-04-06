import os
import warnings
import logging
import csv
from typing import Union

import pandas as pd
import numpy as np

from smac.runhistory.runhistory import RunHistory, DataOrigin
from smac.optimizer.objective import average_cost, _cost
from smac.utils.io.input_reader import InputReader
from smac.tae.execute_ta_run import StatusType

from ConfigSpace import Configuration, ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter
from ConfigSpace.read_and_write import pcs

__author__ = "Joshua Marben"
__copyright__ = "Copyright 2018, ML4AAD"
__license__ = "3-clause BSD"
__maintainer__ = "Joshua Marben"
__email__ = "joshua.marben@neptun.uni-freiburg.de"

class CSV2RH(object):
    def __init__(self):
        pass

    def read_csv_to_rh(self, data,
                       pcs:Union[None, str, ConfigurationSpace]=None,
                       configurations:Union[None, str, dict]=None,
                       train_inst:Union[None, str, list]=None,
                       test_inst:Union[None, str, list]=None,
                       instance_features:Union[None, str, dict]=None,
                       logger=None,
                       seed=42):
        """ Interpreting a .csv-file as runhistory.
        Expecting file with headers:
            p_parameter1, p_parameter2, ...,
            i_instancefeature1 (o), i_instancefeature2 (o), ...,
            seed (o), cost, time (o), status (o)

        Parameters:
        -----------
        csv_path: str
            relative path to csv-file from executing directory
        pcs: Union

        Returns:
        --------
        rh: RunHistory
            runhistory with all the runs from the csv-file
        """
        self.logger = logging.getLogger('cave.utils.csv2rh')
        self.input_reader = InputReader()
        self.pcs = pcs
        self.train_inst = input_reader.read_instance_file(train_inst) if type(train_inst) == str else train_inst
        self.test_inst =  input_reader.read_instance_file(test_inst) if type(test_inst) == str else test_inst
        self.instance_features = input_reader.read_instance_features_file(instance_features) if type(instance_features) == str else instance_features
        self.id_to_config = configurations if configurations else {}
        self.cs = None

        # Read in data
        if isinstance(data, str):
            csv_path = data
            with open(csv_path, 'r') as csv_file:
                csv_data = list(csv.reader(csv_file, delimiter=',',
                                           skipinitialspace=True))
            header, csv_data = csv_data[0], np.array([csv_data[1:]])[0]
            self.data = pd.DataFrame(csv_data, columns=header)
            self.data = self.data.apply(pd.to_numeric, errors='ignore')
            self.logger.debug("Headers: " + str(list(self.data.columns.values)))
        else:
            self.data = data

        # Expecting header as described in docstring
        valid_values = ['seed', 'cost', 'time', 'status', 'config_id', 'instance_id']
        if not len(self.data.columns) == len(set(self.data.columns)):
            raise ValueError("Detected a duplicate in the columns of the "
                             "csv-file \"%s\"." % csv_path)
        for column_name in self.data.columns:
            if not (column_name.lower() in valid_values or
                    column_name.startswith('p_') or  # parameter
                    column_name.startswith('i_')):  # instance feature
                raise ValueError("%s not a legal column name in %s." % column_name, csv_path)

        self._complete_configs()
        self._complete_instances()
        self.logger.debug("Found: seed=%s, cost=%s, time=%s, status=%s",
                          'seed' in self.data.columns, 'cost' in self.data.columns,
                          'time' in self.data.columns, 'status' in self.data.columns)

        # Create RunHistory
        rh = RunHistory(average_cost)
        def add_to_rh(row):
            new_status = self._interpret_status(row['status']) if 'status' in row else StatusType.SUCCESS
            rh.add(config=self.id_to_config[row['config_id']],
                   cost=row['cost'],
                   time=row['time'] if 'time' in row else -1,
                   status=new_status,
                   instance_id=row['instance_id'] if 'instance_id' in row else None,
                   seed=row['seed'] if 'seed' in row else None,
                   additional_info=None,
                   origin=DataOrigin.INTERNAL)

        self.data.apply(add_to_rh, axis=1)
        return rh

    def get_cs(self):
        if self.pcs and type(self.pcs) == ConfigurationSpace:
            cs = self.pcs
        elif self.pcs and type(self.pcs) == str:
            parameters = [p[2:] for p in self.data.columns if p.startswith('p_')]
            with open(self.pcs, 'r') as fh:
                cs = pcs.read(fh)
            diff = set(cs.get_hyperparameter_names()).symmetric_difference(set(parameters))
            if not len(diff) == 0:
                raise ValueError("Comparing parameters from \"%s\" and csv. "
                                 "In pcs but not in csv: [%s]. "
                                 "In csv but not in pcs: [%s]. " % (self.pcs,
                                 str(set(cs.get_hyperparameter_names()).difference(set(parameters))),
                                 str(set(parameters).difference(set(cs.get_hyperparameter_names())))))
        else:
            parameters = [p for p in self.data.columns if p.startswith('p_')]
            warnings.warn("No parameter configuration space (pcs) provided! "
                          "Interpreting all parameters as floats. This might lead "
                          "to suboptimal analysis.", RuntimeWarning)
            minima = self.data.min()  # to define ranges of hyperparameters
            maxima = self.data.max()
            cs = ConfigurationSpace(seed=42)
            for p in parameters:
                cs.add_hyperparameter(UniformFloatHyperparameter(p[2:],
                                      lower=minima[p] - 1, upper=maxima[p] + 1))
        self.cs = cs

    def _interpret_status(self, status):
        if status.upper() in ["SAT", "UNSAT", "SUCCESS"]:
            status = StatusType.SUCCESS
        elif status.upper() in ["TIMEOUT"]:
            status = StatusType.TIMEOUT
        elif status.upper() in ["CRASHED"]:
            status = StatusType.CRASHED
        elif status.upper() in ["ABORT"]:
            status = StatusType.ABORT
        elif status.upper() in ["MEMOUT"]:
            status = StatusType.MEMOUT
        else:
            logger.warning("Could not parse %s as a status. Valid values are: "
                           "[SUCCESS, TIMEOUT, CRASHED, ABORT, MEMOUT]. "
                           "Treating as CRASHED run.")
            status = StatusType.CRASHED
        return status

    def _complete_configs(self):
        """Either config is None or parameters an empty list.
        After completion, every unique configuration in the data will have a
        corresponding id. If specified via parameters, they will be replaced by
        the config-id.

        Parameters:
        -----------
        config_id: bool
            if not False, column_id-column present,
            read in configurations from somewhere
        """
        self.config_to_id = {}
        parameters = [p for p in self.data.columns if p.startswith('p_')]
        if 'config_id' in self.data.columns and len(parameters) > 0:
            raise ValueError("Define configs either via \"p_\" or \"config\" in header, not both.")
        elif 'config_id' in self.data.columns:
            if not self.id_to_config:
                raise ValueError("When defining configs with \"config_id\" "
                                 "in header, you need to provide the argument "
                                 "\"configurations\" to the CSV2RH-object.")
            elif isinstance(self.id_to_config, str):
                # Read in configs from csv
                with open(self.id_to_config, 'r') as csv_file:
                    config_data = list(csv.reader(csv_file, delimiter=',',
                                               skipinitialspace=True))
                header, config_data = config_data[0], np.array([config_data[1:]])[0]
                config_data = pd.DataFrame(config_data, columns=header)
                config_data.set_index('CONFIG_ID', inplace=True)
                config_data = config_data.apply(pd.to_numeric, errors='ignore')
                parameters = ['p_' + p for p in config_data.columns]
                # Create and fill in p_-columns for cs-creation
                for p in parameters:
                    self.data[p] = 0
                def fill_parameters(row):
                    for p in parameters:
                        row[p] = config_data.loc[row['config_id'], p[2:]]
                    return row
                self.data = self.data.apply(fill_parameters, axis=1)
                self.get_cs()
                self.id_to_config = {}
                for index, row in config_data.iterrows():
                    self.id_to_config[index] = Configuration(self.cs,
                            values={name[2:] : value for name, value in
                                             zip(parameters, row)})
                self.config_to_id = {conf : name for name, conf in
                        self.id_to_config.items()}
            elif isinstance(self.id_to_config, dict):
                self.config_to_id = {conf : name for name, conf in
                        self.id_to_config.items()}
                self.get_cs()
        elif parameters:
            # Add new column for config-ids
            self.data['config_id'] = -1
            # Configs are defined via individual "p_"-header-columns
            self.get_cs()
            def add_config(row):
                config = Configuration(self.cs, values={p[2:] : float(row[p]) for p in
                                                   parameters})
                if not config in self.config_to_id.keys():
                    # New index (unseen config)
                    new_id = len(self.config_to_id)
                    self.config_to_id[config] = new_id
                    self.id_to_config[new_id] = config
                row['config_id'] = self.config_to_id[config]
                return row
            self.data = self.data.apply(add_config, axis=1)
        else:
            raise ValueError("Define configs either via \"p_\" or \"config_id\" in header.")

    def _complete_instances(self):
        self.id_to_inst_feats = {}
        self.inst_feats_to_id = {}
        inst_feats = [i for i in self.data.columns if i.startswith('i_')]
        if inst_feats and 'instance_id' in self.data.columns:
            raise ValueError("Define instances either via \"i_\" or \"instance_id\" in header, not both.")
        elif 'instance_id' in self.data.columns:
            if self.instance_features:
                inst_feats = self.instance_features[0]
                self.id_to_inst_feats = {i : tuple(feat) for i, feat in self.instance_features[1]}
                self.inst_feats_to_id = {feat : i for i, feat in self.id_to_inst_feats}
            else:
                self.logger.warning("Instances defined but no instance features available.")
        elif inst_feats:
            # Add new column for instance-ids
            self.data['instance_id'] = -1
            # Instances are defined via individual "i_"-header-columns
            def add_instance(row):
                features = tuple([row[idx] for idx in inst_feats])
                if not features in self.inst_feats_to_id.keys():
                    new_id = len(self.inst_feats_to_id)
                    self.inst_feats_to_id[features] = new_id
                    self.id_to_config[new_id] = features
                row['instance_id'] = self.inst_feats_to_id[features]
                return row
            self.data = self.data.apply(add_instance, axis=1)
        else:
            self.logger.info("No instances detected. Define instances either via \"i_\" or \"instance_id\" in header.")
