
import unittest

from cave.reader.runs_container import RunsContainer
from cave.reader.configurator_run import ConfiguratorRun


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
        folders = ["examples/bohb"]
        rc = RunsContainer(folders, file_format="BOHB")

        print(rc.get_aggregated(True, True))
        print(rc.get_aggregated(True, False))
        print(rc.get_aggregated(False, True))
        print(rc.get_aggregated(False, False))


        #folder = "test/test_files/test_reader/SMAC2/run-1"
        #ta_exec_dir = "test/test_files/test_reader/SMAC2/run-1/smac-output/aclib/state-run1/"
        #cr = ConfiguratorRun(folder, ta_exec_dir, file_format="SMAC2", validation_format=None)
        #self.assertEqual(len(cr.original_runhistory.data), 99)
        #self.assertEqual(len(cr.original_runhistory.get_all_configs()), 43)
        #self.assertIsNone(cr.validated_runhistory)
        #self.assertEqual(len(cr.combined_runhistory.data), 99)
        #self.assertEqual(len(cr.combined_runhistory.get_all_configs()), 43)

        #cr = ConfiguratorRun(folder, ta_exec_dir, file_format="SMAC2", validation_format="SMAC2")
        #self.assertEqual(len(cr.original_runhistory.data), 99)
        #self.assertEqual(len(cr.original_runhistory.get_all_configs()), 43)
        #self.assertEqual(len(cr.validated_runhistory.data), 27)
        #self.assertEqual(len(cr.validated_runhistory.get_all_configs()), 3)
        #self.assertEqual(len(cr.combined_runhistory.data), 126)
        #self.assertEqual(len(cr.combined_runhistory.get_all_configs()), 45)
