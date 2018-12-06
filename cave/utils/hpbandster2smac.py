import os
import logging
import tempfile
import itertools

from ConfigSpace.read_and_write import json as pcs_json
from ConfigSpace.read_and_write import pcs_new
from ConfigSpace.configuration_space import Configuration, ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter
from smac.tae.execute_ta_run import StatusType
from smac.runhistory.runhistory import RunHistory
from smac.optimizer.objective import average_cost
from smac.scenario.scenario import Scenario
from smac.stats.stats import Stats
from smac.utils.io.output_writer import OutputWriter
from smac.utils.io.traj_logging import TrajLogger, TrajEntry

class HpBandSter2SMAC(object):

    def __init__(self):
        self.logger = logging.getLogger(self.__module__ + '.' + self.__class__.__name__)

    def convert(self, folder):
        try:
            from hpbandster.core.result import Result as HPBResult
            from hpbandster.core.result import logged_results_to_HBS_result
        except ImportError as e:
            raise ImportError("To analyze BOHB-data, please install hpbandster (e.g. `pip install hpbandster`)")

        result = logged_results_to_HBS_result(folder)

        # backup_cs is a list with alternative interpretations of the configspace-file (if it's a .pcs-file)
        cs, backup_cs = self.load_configspace(folder)

        # Using temporary files for the intermediate smac-result-like format
        tmp_dir = tempfile.mkdtemp()
        paths = list(self.hpbandster2smac(result, cs, backup_cs, tmp_dir).values())
        return result, paths

    def load_configspace(self, folder):
        """Will try to load the configspace. If it's a pcs-file, backup_cs will be a list containing all possible
        combinations of interpretation for Categoricals. If this issue will be fixed, we can drop this procedure."""
        cs_fn_json = os.path.join(folder, 'configspace.json')
        cs_fn_pcs = os.path.join(folder, 'configspace.pcs')
        if os.path.exists(cs_fn_json):
            with open(cs_fn_json, 'r') as fh:
                cs = pcs_json.read(fh.read())
                backup_cs = []
            self.logger.debug("Detected and loaded \"%s\". No backup-cs necessary", cs_fn_json)
        elif os.path.exists(cs_fn_pcs):
            with open(cs_fn_pcs, 'r') as fh:
                cs = pcs_new.read(fh.readlines())
            # Create alternative interpretations
            categoricals = [hp for hp in cs.get_hyperparameters() if isinstance(hp, CategoricalHyperparameter)]
            non_categoricals = [hp for hp in cs.get_hyperparameters() if not isinstance(hp, CategoricalHyperparameter)]

            def _get_interpretations(choices):
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
            backup_cs = []
            for combi in combinations:
                bcs = ConfigurationSpace()
                for hp in non_categoricals:
                    bcs.add_hyperparameter(hp)
                for name, choices in zip([hp.name for hp in categoricals], combi):
                    bcs.add_hyperparameter(CategoricalHyperparameter(name, choices))
                bcs.add_conditions(cs.get_conditions())
                backup_cs.append(bcs)

            self.logger.debug("Sampled %d interpretations of \"%s\"", len(backup_cs), cs_fn_pcs)
            self.logger.debug(choices_per_cat)
        else:
            raise ValueError("Missing pcs-file at '%s.[pcs|json]'!" % os.path.join(folder, 'configspace'))
        return cs, backup_cs


    def _get_config(self, config_id, id2config, cs):
        config = Configuration(cs, id2config[config_id]['config'])
        try:
            model_based_pick = id2config[config_id]['config_info']['model_based_pick']
            config.origin = 'Model based pick' if model_based_pick else 'Random'
        except KeyError:
            self.logger.debug("No origin for config!", exc_info=True)
        return config

    def hpbandster2smac(self, result, cs: ConfigurationSpace, backup_cs, output_dir: str):
        """Reading hpbandster-result-object and creating RunHistory and
        trajectory...
        treats each budget as an individual 'smac'-run, creates an
        output-directory with subdirectories for each budget.

        Parameters
        ----------
        result: hpbandster.core.result.Result
            bohb's result-object
        cs: ConfigurationSpace
            the configuration space
        backup_cs: List[ConfigurationSpace]
            if loading a configuration fails, try configspaces from this list until succeed
        output_dir: str
            the output-dir to save the smac-runs to
        """
        # Create runhistories (one per budget)
        id2config_mapping = result.get_id2config_mapping()
        budget2rh = {}
        for run in result.get_all_runs():
            if not run.budget in budget2rh:
                budget2rh[run.budget] = RunHistory(average_cost)
            rh = budget2rh[run.budget]

            # Load config...
            try:
                config = self._get_config(run.config_id, id2config_mapping, cs)
            except ValueError as err:
                self.logger.debug("Loading configuration failed... trying alternatives", exc_info=1)
                for bcs in backup_cs:
                    try:
                        config = self._get_config(run.config_id, id2config_mapping, bcs)
                        cs = bcs
                        break
                    except ValueError:
                        self.logger.debug("", exc_info=1)
                        pass
                else:
                    self.logger.debug("None of the alternatives worked...")
                    raise ValueError("Your configspace seems to be corrupt. If you use floats (or mix up ints, bools and strings) as categoricals, "
                                     "please consider using the .json-format, as the .pcs-format cannot recover the type "
                                     "of categoricals. Otherwise please report this to "
                                     "https://github.com/automl/CAVE/issues (and attach the debug.log)")

            if run.loss is not None:
                rh.add(config=config,
                       cost=run.loss,
                       time=run.time_stamps['finished'] - run.time_stamps['started'],
                       status=StatusType.SUCCESS,
                       seed=0,
                       additional_info={'info' : run.info, 'timestamps': run.time_stamps})

        # Write to disk
        budget2path = {}  # paths to individual budgets
        for b, rh in budget2rh.items():
            output_path = os.path.join(output_dir, 'budget_' + str(b))
            budget2path[b] = output_path

            scenario = Scenario({'run_obj' : 'quality', 'cs' : cs, 'output_dir' : output_dir})
            scenario.output_dir_for_this_run = output_path
            scenario.write()
            rh.save_json(fn=os.path.join(output_path, 'runhistory.json'))

            with open(os.path.join(output_path, 'configspace.json'), 'w') as fh:
                fh.write(pcs_json.write(cs))

            self.get_trajectory(result, output_path, scenario, rh, budget=b)

        return budget2path

    def get_trajectory(self, result, output_path, scenario, rh, budget=None):
        """
        If budget is specified, get trajectory for only that budget. Else use hpbandster's averaging.
        """
        cs = scenario.cs
        id2config_mapping = result.get_id2config_mapping()
        if budget:
            traj_dict = self.get_incumbent_trajectory_for_budget(result, budget)
        else:
            traj_dict = result.get_incumbent_trajectory()
        if not output_path:
            output_path = tempfile.mkdtemp()
        traj_logger = TrajLogger(output_path, Stats(scenario))
        for config_id, time, budget, loss in zip(traj_dict['config_ids'], traj_dict['times_finished'], traj_dict['budgets'], traj_dict['losses']):
            incumbent = self._get_config(config_id, id2config_mapping, cs)
            try:
                incumbent_id = rh.config_ids[incumbent]
            except KeyError as e:
                # This config was not evaluated on this budget, just skip it
                continue
            except:
                raise
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
        Returns the best configurations over time

        Parameters
        ----------
        budget: string
            TODO
        result: Result
            result object with runs

        Returns
        -------
            dict:
                dictionary with all the config IDs, the times the runs
                finished, their respective budgets, and corresponding losses
        """
        all_runs = result.get_all_runs(only_largest_budget=False)

        #if not all_budgets:
        #    all_runs = list(filter(lambda r: r.budget==res.HB_config['max_budget'], all_runs))

        all_runs.sort(key=lambda r: (r.budget, r.time_stamps['finished']))

        #self.logger.debug("all runs %s", str(all_runs))

        return_dict = { 'config_ids' : [],
                        'times_finished': [],
                        'budgets'    : [],
                        'losses'     : [],
        }

        current_incumbent = float('inf')
        incumbent_budget = result.HB_config['min_budget']

        for r in all_runs:
            if r.loss is None: continue
            if r.budget != budget: continue

            new_incumbent = False

            if r.loss < current_incumbent:
                new_incumbent = True

            if new_incumbent:
                current_incumbent = r.loss

                return_dict['config_ids'].append(r.config_id)
                return_dict['times_finished'].append(r.time_stamps['finished'])
                return_dict['budgets'].append(r.budget)
                return_dict['losses'].append(r.loss)

        if current_incumbent != r.loss:
            r = all_runs[-1]

            return_dict['config_ids'].append(return_dict['config_ids'][-1])
            return_dict['times_finished'].append(r.time_stamps['finished'])
            return_dict['budgets'].append(return_dict['budgets'][-1])
            return_dict['losses'].append(return_dict['losses'][-1])


        return (return_dict)
