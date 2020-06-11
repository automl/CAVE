import os
import shutil
import tempfile
from collections import OrderedDict

from ConfigSpace.read_and_write import json as pcs_json
from smac.runhistory.runhistory import RunHistory
from smac.scenario.scenario import Scenario
from smac.stats.stats import Stats
from smac.utils.io.input_reader import InputReader
from smac.utils.io.traj_logging import TrajLogger

from cave.reader.base_reader import changedir
from cave.reader.conversion.base_converter import BaseConverter
from cave.reader.conversion.csv2rh import CSV2RH
from cave.utils.helpers import get_folder_basenames
from cave.utils.io import load_config_csv, load_csv_to_pandaframe


class CSV2SMAC(BaseConverter):

    def convert(self, folders, ta_exec_dirs=None, output_dir=None, converted_dest='converted_input_data'):

        self.logger.debug("Converting CSV-data to SMAC3-data. Called with: folders=%s, ta_exec_dirs=%s, output_dir=%s, "
                          "converted_dest=%s", str(folders), str(ta_exec_dirs), str(output_dir), str(converted_dest))

        # Using temporary files for the intermediate smac-result-like format if no output_dir specified
        if not output_dir:
            output_dir = tempfile.mkdtemp()
            self.logger.debug("Temporary directory for intermediate SMAC3-results: %s", output_dir)
        if ta_exec_dirs is None or len(ta_exec_dirs) == 0:
            ta_exec_dirs = ['.']
        if len(ta_exec_dirs) != len(folders):
            ta_exec_dirs = [ta_exec_dirs[0] for _ in folders]

        #####################
        # Actual conversion #
        #####################
        folder_basenames = get_folder_basenames(folders)
        result = OrderedDict()
        for f, f_base, ta_exec_dir in zip(folders, folder_basenames, ta_exec_dirs):  # Those are the parallel runs
            converted_folder_path = os.path.join(output_dir, converted_dest, f_base)
            self.logger.debug("Processing folder=%s, f_base=%s, ta_exec_dir=%s. Saving to %s.",
                              f, f_base, ta_exec_dir, converted_folder_path)
            if not os.path.exists(converted_folder_path):
                self.logger.debug("%s doesn't exist. Creating...", converted_folder_path)
                os.makedirs(converted_folder_path)

            # Get and write scenario # (todo: enhancement: make scenario-file optional (build from scratch))
            scenario_file_path = os.path.join(converted_folder_path, 'scenario.txt')
            scenario = self.get_scenario(f, ta_exec_dir=ta_exec_dir, out_path=scenario_file_path)

            # Read Configuration Space
            config_space = scenario.cs
            #config_space = self.load_configspace(f)
            scenario.paramfile = os.path.join(converted_folder_path, 'configspace.json')
            with open(scenario.paramfile, 'w') as new_file:
                new_file.write(pcs_json.write(config_space))

            # Read runhistory.csv and write runhistory.json(s)
            runhistory = self.get_runhistory(f, scenario, 'runhistory.csv')
            runhistory.save_json(os.path.join(converted_folder_path, 'runhistory.json'))
            try:
                validated_runhistory = self.get_runhistory(f, scenario, 'validated_runhistory.csv')
                validated_runhistory.save_json(os.path.join(converted_folder_path, 'validated_runhistory.json'))
            except FileNotFoundError:
                validated_runhistory = None
                self.logger.debug("No file detected at \"%s\"", os.path.join(f, 'validated_runhistory.csv'))

            # Read trajectory. # (todo: enhancement: make trajectory-file (read it from runhistory?))
            trajectory = self.get_trajectory(f, config_space, scenario, converted_folder_path)

            # After (possibly) changing paths and options (or creating the object), (over)write to new location
            scenario.output_dir_for_this_run = converted_folder_path
            scenario.write()

            result[f] = {
                'new_path' : converted_folder_path,
                'config_space' : config_space,
                'runhistory' : runhistory,
                'validated_runhistory' : validated_runhistory,
                'scenario' : scenario,
                'trajectory' : trajectory,
            }

        return result

    def get_runhistory(self, folder, scenario, filename='runhistory.csv'):
        """Reads runhistory in csv-format:

        +--------------------+--------------------+------+------+------+--------+----------+
        |      config_id     |  instance_id       | cost | time | seed | status | (budget) |
        +====================+====================+======+======+======+========+==========+
        | name of config 1   | name of instance 1 | ...  |  ... | ...  |  ...   | ...      |
        +--------------------+--------------------+------+------+------+--------+----------+
        |         ...        |          ...       | ...  |  ... | ...  |  ...   |  ...     |
        +--------------------+--------------------+------+------+------+--------+----------+

        where config_id and instance_id can also be replaced by columns for the
        individual parameters/instance features

        Sideeffect
        ----------
        Sets self.id_to_config (dict)

        Returns
        -------
        rh: RunHistory
            runhistory
        """
        cs = scenario.cs
        rh_fn = os.path.join(folder, filename)
        if not os.path.exists(rh_fn):
            raise FileNotFoundError("Specified format is \'CSV\', but no \'%s\'-file could be found "
                                    "in \'%s\'" % (filename, folder))
        self.logger.debug("Load Runhistory as csv from %s", rh_fn)
        configs_fn = os.path.join(folder, 'configurations.csv')
        if os.path.exists(configs_fn):
            self.logger.debug("Found \'configurations.csv\' in %s." % folder)
            self.id_to_config = load_config_csv(configs_fn, cs, self.logger)[1]
        else:
            self.logger.debug("No \'configurations.csv\' in %s." % folder)
            self.id_to_config = {}

        rh = CSV2RH().read_csv_to_rh(rh_fn,
                                     cs=cs,
                                     id_to_config=self.id_to_config,
                                     train_inst=scenario.train_insts,
                                     test_inst=scenario.test_insts,
                                     instance_features=scenario.feature_dict,
                                     )
        if not self.id_to_config:
            self.id_to_config = rh.ids_config

        return rh

    def get_trajectory(self, folder, cs, scenario, output_path):
        """Reads `folder/trajectory.csv`, expected format:

        +----------+------+----------------+-------------+-----------+
        | cpu_time | cost | wallclock_time | evaluations | config_id |
        +==========+======+================+=============+===========+
        | ...      | ...  | ...            | ...         | ...       |
        +----------+------+----------------+-------------+-----------+

        or

        +----------+------+----------------+-------------+------------+------------+-----+
        | cpu_time | cost | wallclock_time | evaluations | parameter1 | parameter2 | ... |
        +==========+======+================+=============+============+============+=====+
        | ...      | ...  | ...            | ...         | ...        | ...        | ... |
        +----------+------+----------------+-------------+------------+------------+-----+

        Sideeffect
        ----------
        Writes trajectory to trajectory-file in output-dir

        Returns
        -------
        traj: List[TrajEntry]
            Returns trajectory as list
        """
        traj_fn = os.path.join(folder, 'trajectory.csv')
        if not os.path.exists(traj_fn):
            raise FileNotFoundError("Specified format is \'CSV\', but no \'../trajectory\'"
                                    "-file at \'{}\'.".format(traj_fn))

        csv_data = load_csv_to_pandaframe(traj_fn, self.logger, apply_numeric=False)
        traj_logger = TrajLogger(output_path, Stats(scenario))

        csv_data, configs = CSV2RH().extract_configs(csv_data, cs, self.id_to_config)
        def add_to_traj(row):
            """ Adds a new entry to the trajectory (and logs it to the trajectory file)"""
            new_entry = {
                'cpu_time' : float(row['cpu_time']),
                'total_cpu_time' : None,
                "wallclock_time" : float(row['wallclock_time']),
                "evaluations" : int(row['evaluations']),
                "cost" : float(row["cost"]),
                "incumbent" : self.id_to_config[row["config_id"]],
                "budget": float(row["budget"]) if "budget" in row else 0,
            }
            traj_logger.trajectory.append(new_entry)
            traj_logger._add_in_alljson_format(train_perf=new_entry['cost'],
                                               incumbent_id=row['config_id'],
                                               incumbent=new_entry['incumbent'],
                                               budget=new_entry['budget'],
                                               ta_time_used=new_entry['cpu_time'],
                                               wallclock_time=new_entry['wallclock_time'],
                                               )
        csv_data.apply(add_to_traj, axis=1)

        return traj_logger.trajectory

    def get_scenario(self, path, ta_exec_dir=None, out_path=None):
        run_1_existed = os.path.exists('run_1')
        if ta_exec_dir is None:
            ta_exec_dir = '.'
        in_reader = InputReader()
        # Create Scenario
        scen_fn = os.path.join(path, 'scenario.txt')
        scen_dict = in_reader.read_scenario_file(scen_fn)
        scen_dict['output_dir'] = out_path
        with changedir(ta_exec_dir):
            self.logger.debug("Creating scenario from \"%s\"", ta_exec_dir)
            scen = Scenario(scen_dict)

        if (not run_1_existed) and os.path.exists('run_1'):
            shutil.rmtree('run_1')
        return scen

    @classmethod
    def check_for_files(cls, path):
        """ Returns True if all files needed for CSV formatted results are detected in target folder """
        if (os.path.isfile(os.path.join(path, 'scenario.txt'))
            and os.path.isfile(os.path.join(path, 'runhistory.csv'))
            and os.path.isfile(os.path.join(path, 'trajectory.csv'))
        ):
            return True
        return False