import os
import logging
import tempfile

from ConfigSpace.read_and_write import pcs_new
from smac.configspace import Configuration, ConfigurationSpace
from smac.tae.execute_ta_run import StatusType
from smac.runhistory.runhistory import RunHistory
from smac.optimizer.objective import average_cost
from smac.scenario.scenario import Scenario
from smac.stats.stats import Stats
from smac.utils.io.output_writer import OutputWriter
from smac.utils.io.traj_logging import TrajLogger

class HpBandSter2SMAC(object):

    def __init__(self):
        self.logger = logging.getLogger(self.__module__ + '.' + self.__class__.__name__)

    def convert(self, folder):
        try:
            from hpbandster.core.result import Result as HPBResult
            from hpbandster.core.result import logged_results_to_HB_result
        except ImportError as e:
            raise ImportError("To analyze BOHB-data, please install hpbandster (e.g. `pip install hpbandster`)")

        result = logged_results_to_HB_result(folder)

        cs_fn = os.path.join(folder, 'configspace.pcs')
        if not os.path.exists(cs_fn):
            raise ValueError("Missing pcs-file at '%s'!" % cs_fn)
        with open(cs_fn, 'r') as fh:
            cs = pcs_new.read(fh.readlines())

        # Using temporary files for the intermediate smac-result-like format
        tmp_dir = tempfile.mkdtemp()
        paths = list(self.hpbandster2smac(result, cs, tmp_dir).values())
        return result, paths

    def hpbandster2smac(self, result, cs: ConfigurationSpace, output_dir: str):
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
            rh.add(config=Configuration(cs, id2config_mapping[run.config_id]['config']),
                   cost=run.loss,
                   time=run.time_stamps['finished'] - run.time_stamps['started'],
                   status=StatusType.SUCCESS,
                   seed=0,
                   additional_info={'info' : run.info})

        # Write to disk
        budget2path = {}  # paths to individual budgets
        for b, rh in budget2rh.items():
            output_path = os.path.join(output_dir, 'budget_' + str(b))
            budget2path[b] = output_path

            scenario = Scenario({'run_obj' : 'quality', 'cs' : cs})
            scenario.output_dir_for_this_run = output_path
            scenario.write()
            rh.save_json(fn=os.path.join(output_path, 'runhistory.json'))

            # trajectory
            traj_dict = result.get_incumbent_trajectory()
            traj_logger = TrajLogger(output_path, Stats(scenario))
            for config_id, time, budget, loss in zip(traj_dict['config_ids'], traj_dict['times_finished'], traj_dict['budgets'], traj_dict['losses']):
                incumbent = Configuration(cs, id2config_mapping[config_id]['config'])
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
                traj_logger._add_in_old_format(train_perf, incumbent_id, incumbent,
                                               ta_time_used, wallclock_time)
                traj_logger._add_in_aclib_format(train_perf, incumbent_id, incumbent,
                                                 ta_time_used, wallclock_time)

        return budget2path
