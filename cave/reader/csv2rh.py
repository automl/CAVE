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
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, CategoricalHyperparameter
from ConfigSpace.read_and_write import pcs

from cave.utils.io import load_csv_to_pandaframe

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
        self.train_inst = input_reader.read_instance_file(train_inst) if type(train_inst) == str else train_inst
        self.test_inst =  input_reader.read_instance_file(test_inst) if type(test_inst) == str else test_inst
        feature_names = []  # names of instance-features
        if type(instance_features) == str:
            names, feats = input_reader.read_instance_features_file(instance_features)
            feature_names = names
            self.instance_features = feats
        else:
            self.instance_features = instance_features

        # Read in data
        if isinstance(data, str):
            self.logger.debug("Detected path for csv-file (\'%s\')", data)
            self.data = load_csv_to_pandaframe(data, self.logger)
        else:
            self.data = data

        # Expecting header as described in docstring
        self.valid_values = ['seed', 'cost', 'time', 'status', 'config_id', 'instance_id']
        if not len(self.data.columns) == len(set(self.data.columns)):
            raise ValueError("Detected a duplicate in the columns of the "
                             "csv-file \"%s\"." % csv_path)

        if isinstance(configurations, str):
            parameters, self.id_to_config = self._load_config_csv(configurations, pcs)
        elif isinstance(configurations, dict):
            self.id_to_config = configurations
            parameters = np.random.choice(list(self.id_to_config.values())).configuration_space.get_hyperparameter_names()
        else:
            self.id_to_config = {}
            parameters = [c for c in self.data.columns if not
                          (c.lower() in self.valid_values)]
        if not feature_names:
            feature_names = [c for c in self.data.columns if
                             not c.lower() in self.valid_values and
                             not c in parameters]
        self.cs = self.get_cs(pcs, parameters)

        self._complete_configs()
        self._complete_instances(feature_names)
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

    def get_cs(self, cs, parameters=None):
        if cs and isinstance(cs, ConfigurationSpace):
            cs = cs
        elif cs and isinstance(cs, str):
            self.logger.debug("Reading PCS from %s", cs)
            with open(cs, 'r') as fh:
                cs = pcs.read(fh)
        elif parameters:
            # TODO use from pyimp after https://github.com/automl/ParameterImportance/issues/72 is implemented
            warnings.warn("No parameter configuration space (pcs) provided! "
                          "Interpreting all parameters as floats. This might lead "
                          "to suboptimal analysis.", RuntimeWarning)
            self.logger.debug("Interpreting as parameters: %s", parameters)
            minima = self.data.min()  # to define ranges of hyperparameters
            maxima = self.data.max()
            cs = ConfigurationSpace(seed=42)
            for p in parameters:
                cs.add_hyperparameter(UniformFloatHyperparameter(p,
                                      lower=minima[p] - 1, upper=maxima[p] + 1))
        else:
            raise ValueError("No pcs-file provided, parameters could not be interpreted.")
        return cs

    def _interpret_status(self, status):
        status = status.strip().upper()
        if status in ["SAT", "UNSAT", "SUCCESS"]:
            status = StatusType.SUCCESS
        elif status in ["TIMEOUT"]:
            status = StatusType.TIMEOUT
        elif status in ["CRASHED"]:
            status = StatusType.CRASHED
        elif status in ["ABORT"]:
            status = StatusType.ABORT
        elif status in ["MEMOUT"]:
            status = StatusType.MEMOUT
        else:
            logger.warning("Could not parse %s as a status. Valid values are: "
                           "[SUCCESS, TIMEOUT, CRASHED, ABORT, MEMOUT]. "
                           "Treating as CRASHED run.", status)
            status = StatusType.CRASHED
        return status

    def _complete_configs(self):
        """Either config is None or parameters an empty list.
        After completion, every unique configuration in the data will have a
        corresponding id. If specified via parameters, they will be replaced by
        the config-id.
        """
        self.config_to_id = {}
        if 'config_id' in self.data.columns:
            if not self.id_to_config:
                raise ValueError("When defining configs with \"config_id\" "
                                 "in header, you need to provide the argument "
                                 "\"configurations\" to the CSV2RH-object - "
                                 "either as a dict, mapping the id's to "
                                 "Configurations or as a path to a csv-file "
                                 "containing the necessary information.")
            self.config_to_id = {conf : name for name, conf in
                                 self.id_to_config.items()}
        else:
            self.logger.debug("No \'config_id\'-column detected. Interpreting "
                              "from pcs, if available.")
            # Add new column for config-ids
            self.data['config_id'] = -1
            for idx, row in enumerate(self.data.itertuples()):
                values = {p : getattr(row, p) for p in
                          self.cs.get_hyperparameter_names()}
                values = {k : str(v) if isinstance(self.cs.get_hyperparameter(k),
                                                   CategoricalHyperparameter)
                          else v for k, v in values.items()}
                config = Configuration(self.cs, values=values)
                if not config in self.config_to_id.keys():
                    # New index (unseen config)
                    new_id = len(self.config_to_id)
                    self.config_to_id[config] = new_id
                    self.id_to_config[new_id] = config
                self.data.loc[idx, 'config_id'] = self.config_to_id[config]

    def _complete_instances(self, feature_names):
        self.id_to_inst_feats = {}
        self.inst_feats_to_id = {}
        inst_feats = feature_names
        if 'instance_id' in self.data.columns:
            if self.instance_features:
                self.id_to_inst_feats = {i : tuple(feat) for i, feat in
                                         self.instance_features.items()}
                self.inst_feats_to_id = {feat : i for i, feat in
                                         self.id_to_inst_feats.items()}
            else:
                raise ValueError("Instances defined via \'instance_id\'-column, "
                                 "but no instance features available.")
        elif inst_feats:
            # Add new column for instance-ids
            self.data['instance_id'] = -1
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
            self.logger.info("No instances detected.")
