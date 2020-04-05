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
from smac.utils.io.traj_logging import TrajLogger

from cave.utils.hpbandster_helpers import get_incumbent_trajectory, format_budgets


class HpBandSter2SMAC(object):

    def __init__(self):
        self.logger = logging.getLogger(self.__module__ + '.' + self.__class__.__name__)

    def convert(self, folders, output_dir=None):
        """Convert hpbandster-results into smac-format, aggregating parallel runs along the budgets, so it is treated as
        one run with the same budgets. Throws ValueError when budgets of individual runs dont match.

        WIP: make hpbandsterconversion not aggregate parallel runs

        Parameters
        ----------
        folders: List[str]
            list of parallel hpbandster-runs (folder paths!)
        output_dir: str
            path to CAVE's output-directory

        Returns
        -------
        folder2result: {str : hpbandster.core.result}
            map parallel-run-folder-paths to hpbandster-result in original format
        folder2budgets: {str : {str or int or float : str}}
            map folder to budget to pathpaths to converted data
        """
        try:
            from hpbandster.core.result import Result as HPBResult
            from hpbandster.core.result import logged_results_to_HBS_result
        except ImportError as e:
            raise ImportError("To analyze BOHB-data, please install hpbandster (e.g. `pip install hpbandster`)")

        # Original hpbandster-formatted result-object
        folder2result = OrderedDict([(f, logged_results_to_HBS_result(f)) for f in folders])

        # Get a list with alternative interpretations of the configspace-file (if it's a .pcs-file, for .json-files it's
        # only one element)
        cs_interpretations = self.load_configspace(folders[0])

        # Using temporary files for the intermediate smac-result-like format
        if not output_dir:
            self.logger.debug("New outputdir")
            output_dir = tempfile.mkdtemp()

        # Actual conversion
        folder2budgets = self.hpbandster2smac(folder2result, cs_interpretations, output_dir)

        return folder2result, folder2budgets

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

    def hpbandster2smac(self, folder2result, cs_options, output_dir: str):
        """Reading hpbandster-result-object and creating RunHistory and trajectory...  treats each budget as an
        individual 'smac'-run, creates an output-directory with subdirectories for each budget.

        Parameters
        ----------
        folder2result: Dict(str : hpbandster.core.result.Result)
            folder mapping to bohb's result-objects
        cs_options: list[ConfigurationSpace]
            the configuration spaces. in the best case it's a single element, but for pcs-format we need to guess
            through a list of possible configspaces
        output_dir: str
            the output-dir to save the smac-runs to
        
        Returns
        -------
        folder2budgets: dict(dict(str) - str)
            maps each folder (from parallel execution) to a dict, which in turn maps all budgets of
            the specific parallel execution to their paths
        """
        folder2budgets = OrderedDict()
        self.logger.debug("Loading with %d configspace alternative options...", len(cs_options))
        self.logger.info("Assuming BOHB treats target algorithms as deterministic (and does not re-evaluate)")
        for folder, result in folder2result.items():
            folder2budgets[folder] = OrderedDict()
            self.logger.debug("Budgets for '%s': %s" % (folder, str(result.HB_config['budgets'])))
            ##########################
            # 1. Create runhistory   #
            ##########################
            id2config_mapping = result.get_id2config_mapping()
            skipped = {'None' : 0, 'NaN' : 0}
            budget2rh = OrderedDict()
            for run in result.get_all_runs():
                # Choose runhistory to add run to
                if not run.budget in budget2rh:
                    budget2rh[run.budget] = RunHistory()
                rh = budget2rh[run.budget]

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
                       seed=0,
                       additional_info={'info' : run.info, 'timestamps': run.time_stamps})

            self.logger.debug("Skipped %d None- and %d NaN-loss-values in BOHB-result", skipped['None'], skipped['NaN'])

            ##########################
            # 2. Create all else     #
            ##########################
            formatted_budgets = format_budgets(budget2rh.keys())  # Make budget-names readable [0.021311, 0.031211] to [0.02, 0.03]
            for b, rh in budget2rh.items():
                output_path = os.path.join(output_dir, folder, formatted_budgets[b])
                folder2budgets[folder][b] = output_path

                scenario = Scenario({'run_obj' : 'quality',
                                     'cs' : cs_options[0],
                                     'output_dir' : output_dir,
                                     'deterministic' : True,  # At the time of writing, BOHB is always treating ta's as deterministic
                                     })
                scenario.output_dir_for_this_run = output_path
                scenario.write()

                with open(os.path.join(output_path, 'configspace.json'), 'w') as fh:
                    fh.write(pcs_json.write(cs_options[0]))

                rh.save_json(fn=os.path.join(output_path, 'runhistory.json'))

                self.get_trajectory(folder2result[folder], output_path, scenario, rh, budget=b)

        return folder2budgets

    def get_trajectory(self, result, output_path, scenario, rh, budget=None):
        """
        If budget is specified, get trajectory for only that budget. Else use hpbandster's averaging.

        TODO: would like to be rewritten
        """
        cs = scenario.cs

        if not output_path:
            output_path = tempfile.mkdtemp()

        traj_logger = TrajLogger(output_path, Stats(scenario))
        total_traj_dict = []
        if budget:
            traj_dict = get_incumbent_trajectory(result, [budget])
        else:
            traj_dict = result.get_incumbent_trajectory()

        id2config_mapping = result.get_id2config_mapping()

        for config_id, time, budget, loss in zip(traj_dict['config_ids'],
                                                 traj_dict['times_finished'],
                                                 traj_dict['budgets'],
                                                 traj_dict['losses']):
            incumbent = self._get_config(config_id, id2config_mapping, cs)
            try:
                incumbent_id = rh.config_ids[incumbent]
            except KeyError as e:
                # This config was not evaluated on this budget, just skip it
                continue
            except:
                raise
            total_traj_dict.append({'config_id' : incumbent_id,
                                    'time_finished' : time,
                                    'budget' : budget,
                                    'loss' : loss})

        last_loss = np.inf
        for element in sorted(total_traj_dict, key=lambda x: x['time_finished']):
            incumbent_id = element["config_id"]
            incumbent = rh.ids_config[incumbent_id]
            time = element["time_finished"]
            loss = element["loss"]

            if loss > last_loss:
                continue

            ta_runs = -1
            ta_time_used = -1
            wallclock_time = time
            train_perf = loss
            # add
            traj_logger.trajectory.append({"cpu_time": ta_time_used,
                                           "total_cpu_time": None,  # TODO: fix this
                                           "wallclock_time": wallclock_time,
                                           "evaluations": ta_runs,
                                           "cost": train_perf,
                                           "incumbent": incumbent
                                           })
            traj_logger._add_in_old_format(train_perf, incumbent_id, incumbent, ta_time_used, wallclock_time)
            traj_logger._add_in_aclib_format(train_perf, incumbent_id, incumbent, ta_time_used, wallclock_time)
        return traj_logger.trajectory

    def get_incumbent_trajectory_for_budget(self, result, budget):
        """
        Returns the best configurations over time for a single budget

        Parameters
        ----------
        budget: string
            budget to be considered
        result: Result
            result object with runs

        Returns
        -------
            dict:
                dictionary with all the config IDs, the times the runs
                finished, their respective budgets, and corresponding losses
        """
        if not budget in result.HB_config['budgets']:
            raise ValueError("Budget '{}' (type: {}) does not exist. Choose from {}".format(str(budget), str(type(budget)),
                "[" + ", ".join([str(b) + " (type: " +  str(type(b)) + ")" for b in result.HB_config['budgets']]) + "]"))
