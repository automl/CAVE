import itertools
import logging
import os
import tempfile
from collections import OrderedDict

import numpy as np
from ConfigSpace.configuration_space import Configuration, ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter
from ConfigSpace.read_and_write import json as pcs_json
from ConfigSpace.read_and_write import pcs_new
from smac.runhistory.runhistory import RunHistory
from smac.scenario.scenario import Scenario
from smac.stats.stats import Stats
from smac.tae.execute_ta_run import StatusType
from smac.utils.io.traj_logging import TrajLogger, TrajEntry

from cave.reader.conversion.base_converter import BaseConverter
from cave.utils.helpers import get_folder_basenames
from cave.utils.hpbandster_helpers import get_incumbent_trajectory, format_budgets


class HpBandSter2SMAC(BaseConverter):

    def convert(self, folders, ta_exec_dirs=None, output_dir=None, converted_dest='converted_input_data'):
        try:
            from hpbandster.core.result import Result as HPBResult
            from hpbandster.core.result import logged_results_to_HBS_result
        except ImportError as e:
            raise ImportError("To analyze BOHB-data, please install hpbandster (e.g. `pip install hpbandster`)")

        self.logger.debug("Converting BOHB-data to SMAC3-data. Called with: folders=%s, ta_exec_dirs=%s, output_dir=%s, "
                          "converted_dest=%s", str(folders), str(ta_exec_dirs), str(output_dir), str(converted_dest))

        # Using temporary files for the intermediate smac-result-like format if no output_dir specified
        if not output_dir:
            output_dir = tempfile.mkdtemp()
            self.logger.debug("Temporary directory for intermediate SMAC3-results: %s", output_dir)
        if ta_exec_dirs is None or len(ta_exec_dirs) == 0:
            ta_exec_dirs = ['.']
        if len(ta_exec_dirs) != len(folders):
            ta_exec_dirs = [ta_exec_dirs[0] for _ in folders]


        # Get a list with alternative interpretations of the configspace-file
        # (if it's a .pcs-file, for .json-files it's a length-one-list)
        cs_interpretations = self.load_configspace(folders[0])
        self.logger.debug("Loading with %d configspace alternative options...", len(cs_interpretations))
        self.logger.info("Assuming BOHB treats target algorithms as deterministic (and does not re-evaluate)")

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

            # Original hpbandster-formatted result-object
            hp_result = logged_results_to_HBS_result(f)
            result[f] = self.hpbandster2smac(f, hp_result, cs_interpretations, converted_folder_path)

        return result

    def load_configspace(self, folder):
        """Will try to load the configspace. cs_options will be a list containing all possible
        combinations of interpretation for Categoricals. If this issue will be fixed, we can drop this procedure.

        Parameters
        ----------
        folder: str
            path to folder in which to look for configspace

        Returns
        -------
        cs_options: list[ConfigurationSpace]
            list with possible interpretations for config-space-file. Only contains multiple items if file-format is pcs.
        """
        cs_options = []
        cs_fn_json = os.path.join(folder, 'configspace.json')
        cs_fn_pcs = os.path.join(folder, 'configspace.pcs')

        if os.path.exists(cs_fn_json):
            with open(cs_fn_json, 'r') as fh:
                cs_options = [pcs_json.read(fh.read())]
            self.logger.debug("Detected and loaded \"%s\". No alternative interpretations necessary", cs_fn_json)
        elif os.path.exists(cs_fn_pcs):
            with open(cs_fn_pcs, 'r') as fh:
                cs = pcs_new.read(fh.readlines())
            # Create alternative interpretations
            categoricals = [hp for hp in cs.get_hyperparameters() if isinstance(hp, CategoricalHyperparameter)]
            non_categoricals = [hp for hp in cs.get_hyperparameters() if not isinstance(hp, CategoricalHyperparameter)]

            def _get_interpretations(choices):
                """ Generate different interpretations for critical categorical hyperparameters that are not seamlessly
                supported by pcs-format."""
                result = []
                if set(choices) == {"True", "False"}:
                    result.append([True, False])
                if all([c.isdigit() for c in choices]):
                    result.append([int(c) for c in choices])
                result.append(choices)
                return result

            choices_per_cat = [_get_interpretations(hp.choices) for hp in categoricals]
            combinations = itertools.product(*choices_per_cat)
            self.logger.debug(combinations)
            for combi in combinations:
                bcs = ConfigurationSpace()
                for hp in non_categoricals:
                    bcs.add_hyperparameter(hp)
                for name, choices in zip([hp.name for hp in categoricals], combi):
                    bcs.add_hyperparameter(CategoricalHyperparameter(name, choices))
                bcs.add_conditions(cs.get_conditions())
                cs_options.append(bcs)

            self.logger.debug("Sampled %d interpretations of \"%s\"", len(cs_options), cs_fn_pcs)
        else:
            raise ValueError("Missing pcs-file at '%s.[pcs|json]'!" % os.path.join(folder, 'configspace'))
        return cs_options

    def _get_config(self, config_id, id2config, cs):
        config = Configuration(cs, id2config[config_id]['config'])
        try:
            model_based_pick = id2config[config_id]['config_info']['model_based_pick']
            config.origin = 'Model based pick' if model_based_pick else 'Random'
        except KeyError:
            self.logger.debug("No origin for config (id %s)!" % str(config_id), exc_info=True)
        return config

    def hpbandster2smac(self, folder, result, cs_options, output_dir: str):
        """Reading hpbandster-result-object and creating RunHistory and trajectory...  treats each budget as an
        individual 'smac'-run, creates an output-directory with subdirectories for each budget.

        Parameters
        ----------
        folder: str (path)
            original folder
        result: hpbandster.core.result.Result
            bohb's result-object
        cs_options: list[ConfigurationSpace]
            the configuration spaces. in the best case it's a single element, but for pcs-format we need to guess
            through a list of possible configspaces
        output_dir: str
            the output-dir to save the smac-runs to
        
        Returns
        -------
        converted: dict{
                'config_space' : config_space,
                'runhistory' : runhistory,
                'validated_runhistory' : validated_runhistory,
                'scenario' : scenario,
                'trajectory' : trajectory,
                }
        """
        self.logger.debug("Budgets for '%s': %s" % (folder, str(result.HB_config['budgets'])))
        ##########################
        # 1. Create runhistory   #
        ##########################
        id2config_mapping = result.get_id2config_mapping()
        skipped = {'None' : 0, 'NaN' : 0}
        rh = RunHistory()
        for run in result.get_all_runs():
            # Load config...
            config = None
            while config is None:
                if len(cs_options) == 0:
                    self.logger.debug("None of the alternatives worked...")
                    raise ValueError("Your configspace seems to be corrupt. If you use floats (or mix up ints, bools and strings) as categoricals, "
                                     "please consider using the .json-format, as the .pcs-format cannot recover the type "
                                     "of categoricals. Otherwise please report this to "
                                     "https://github.com/automl/CAVE/issues (and attach the debug.log)")
                try:
                    config = self._get_config(run.config_id, id2config_mapping, cs_options[0])
                except ValueError as err:
                    self.logger.debug("Loading configuration failed... trying %d alternatives" % len(cs_options) - 1, exc_info=1)
                    cs_options = cs_options[1:]  # remove the failing cs-version

            # Filter corrupted loss-values (ignore them)
            if run.loss is None:
                skipped['None'] += 1
                continue
            if np.isnan(run.loss):
                skipped['NaN'] += 1
                continue

            rh.add(config=config,
                   cost=run.loss,
                   time=run.time_stamps['finished'] - run.time_stamps['started'],
                   status=StatusType.SUCCESS,
                   budget=run.budget,
                   seed=0,
                   additional_info={'info' : run.info, 'timestamps': run.time_stamps})

        self.logger.debug("Skipped %d None- and %d NaN-loss-values in BOHB-result", skipped['None'], skipped['NaN'])

        ##########################
        # 2. Create all else     #
        ##########################
        scenario = Scenario({'run_obj' : 'quality',
                             'cs' : cs_options[0],
                             'output_dir' : output_dir,
                             'deterministic' : True,  # At the time of writing, BOHB is always treating ta's as deterministic
                            })
        scenario.output_dir_for_this_run = output_dir
        scenario.write()

        with open(os.path.join(output_dir, 'configspace.json'), 'w') as fh:
            fh.write(pcs_json.write(cs_options[0]))

        rh.save_json(fn=os.path.join(output_dir, 'runhistory.json'))

        trajectory = self.get_trajectory(result, output_dir, scenario, rh)

        return {'new_path' : output_dir,
                'hpbandster_result' : result,
                'config_space' : cs_options[0],
                'runhistory' : rh,
                'validated_runhistory' : None,
                'scenario' : scenario,
                'trajectory' : trajectory,
                }

    def get_trajectory(self, result, output_path, scenario, rh):
        """
        Use hpbandster's averaging.
        """
        cs = scenario.cs

        if not output_path:
            output_path = tempfile.mkdtemp()

        traj_logger = TrajLogger(output_path, Stats(scenario))
        total_traj_dict = []
        traj_dict = result.get_incumbent_trajectory()

        id2config_mapping = result.get_id2config_mapping()

        failed_entries = []
        for config_id, time, budget, loss in zip(traj_dict['config_ids'],
                                                 traj_dict['times_finished'],
                                                 traj_dict['budgets'],
                                                 traj_dict['losses']):
            incumbent = self._get_config(config_id, id2config_mapping, cs)
            try:
                incumbent_id = rh.config_ids[incumbent]
            except KeyError as err:
                self.logger.debug(err)
                failed_entries.append((config_id, incumbent))
                self.logger.debug("Could not load configuration id %d (%s)", config_id, str(incumbent))
                continue
            total_traj_dict.append({'config_id' : incumbent_id,
                                    'time_finished' : time,
                                    'budget' : budget,
                                    'loss' : loss})
        if len(failed_entries) > 0:
            self.logger.warning("Failed to load %d (of %d total) entries from trajectory", len(failed_entries), len(failed_entries) + len(total_traj_dict))

        last_loss = np.inf
        for element in sorted(total_traj_dict, key=lambda x: x['time_finished']):
            incumbent_id = element["config_id"]
            incumbent = rh.ids_config[incumbent_id]
            time = element["time_finished"]
            loss = element["loss"]
            budget = element["budget"]

            if loss > last_loss:
                continue

            ta_runs = -1
            ta_time_used = -1
            wallclock_time = time
            train_perf = loss
            # add to trajectory, imitate `add_entry` method of SMAC's traj_logger
            traj_logger.trajectory.append({
                'cpu_time' : ta_time_used,
                'total_cpu_time' : None,
                "wallclock_time" : wallclock_time,
                "evaluations" : ta_runs,
                "cost" : train_perf,
                "incumbent" : incumbent,
                "budget" : budget
            })
            traj_logger._add_in_alljson_format(train_perf,
                                               incumbent_id,
                                               incumbent,
                                               budget,
                                               ta_time_used,
                                               wallclock_time,
                                               )
        return traj_logger.trajectory
