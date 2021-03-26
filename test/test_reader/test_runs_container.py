import unittest

from cave.reader.configurator_run import ConfiguratorRun
from cave.reader.runs_container import RunsContainer


class TestRunContainer(unittest.TestCase):

    def test_runs_container(self):
        """ test whether runs_container-methods work as expected """
        folders = ["examples/bohb"]
        rc = RunsContainer(folders, file_format="BOHB")

        print(rc.get_all_runs())
        self.assertEqual(rc.get_highest_budget(), 100)
        self.assertEqual(rc.get_budgets(), [6.25, 12.5, 25.0, 50.0, 100.0])
        print(rc.get_runs_for_budget(rc.get_highest_budget()))
        print(rc.get_folders())
        print(rc.get_runs_for_folder(folders[0]))

    def test_runs_aggregation(self):
        """ test whether runs_container-methods work as expected """
        folders = ["examples/smac3/example_output/run_1", "examples/smac3/example_output/run_2"]
        ta_exec_dir = ["examples/smac3"]
        rc = RunsContainer(folders, ta_exec_dirs=ta_exec_dir, file_format="SMAC3")

        self.assertEqual(len(rc["examples/smac3/example_output/run_1"].original_runhistory.data), 271)
        self.assertEqual(len(rc["examples/smac3/example_output/run_1"].original_runhistory.get_all_configs()), 62)
        self.assertEqual(len(rc["examples/smac3/example_output/run_2"].original_runhistory.data), 299)
        self.assertEqual(len(rc["examples/smac3/example_output/run_2"].original_runhistory.get_all_configs()), 63)

        agg = rc.get_aggregated(keep_budgets=True, keep_folders=True)
        self.assertEqual(len(agg), 2)
        agg = rc.get_aggregated(keep_budgets=True, keep_folders=False)
        self.assertEqual(len(agg), 1)
        agg = agg[0]
        self.assertIsInstance(agg, ConfiguratorRun)
        self.assertEqual(len(agg.original_runhistory.data), 570)
        self.assertEqual(len(agg.original_runhistory.get_all_configs()), 124)
        agg = rc.get_aggregated(keep_budgets=False, keep_folders=True)
        self.assertEqual(len(agg), 2)
        agg = rc.get_aggregated(keep_budgets=False, keep_folders=False)
        self.assertEqual(len(agg), 1)
        agg = agg[0]
        self.assertIsInstance(agg, ConfiguratorRun)
        self.assertEqual(len(agg.original_runhistory.data), 570)
        self.assertEqual(len(agg.original_runhistory.get_all_configs()), 124)

    def test_runs_aggregation_bohb(self):
        """ test whether runs_container-methods work as expected """
        # TODO extend to multiple bohb-dirs
        folders = ["examples/bohb"]
        ta_exec_dir = ["."]
        rc = RunsContainer(folders, ta_exec_dirs=ta_exec_dir, file_format="BOHB")

        self.assertEqual(len(rc["examples/bohb"].original_runhistory.data), 256)
