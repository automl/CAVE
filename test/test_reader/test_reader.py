import unittest

from cave.reader.configurator_run import ConfiguratorRun


class TestReader(unittest.TestCase):

    def test_smac3_format(self):
        folder = "examples/smac3/example_output/run_1"
        ta_exec_dir = "examples/smac3/"
        cr = ConfiguratorRun.from_folder(folder, ta_exec_dir, file_format="SMAC3", validation_format=None)
        self.assertEqual(len(cr.original_runhistory.data), 461)
        self.assertEqual(len(cr.original_runhistory.get_all_configs()), 71)
        self.assertIsNone(cr.validated_runhistory)
        self.assertEqual(len(cr.combined_runhistory.data), 461)
        self.assertEqual(len(cr.combined_runhistory.get_all_configs()), 71)

    def test_smac2_format(self):
        """ test whether smac2-format is correctly interpreted """
        folder = "test/test_files/test_reader/SMAC2/run-1"
        ta_exec_dir = "test/test_files/test_reader/SMAC2/run-1/smac-output/aclib/state-run1/"
        cr = ConfiguratorRun.from_folder(folder, ta_exec_dir, file_format="SMAC2", validation_format=None)
        self.assertEqual(len(cr.original_runhistory.data), 99)
        self.assertEqual(len(cr.original_runhistory.get_all_configs()), 43)
        self.assertIsNone(cr.validated_runhistory)
        self.assertEqual(len(cr.combined_runhistory.data), 99)
        self.assertEqual(len(cr.combined_runhistory.get_all_configs()), 43)

        cr = ConfiguratorRun.from_folder(folder, ta_exec_dir, file_format="SMAC2", validation_format="SMAC2")
        self.assertEqual(len(cr.original_runhistory.data), 99)
        self.assertEqual(len(cr.original_runhistory.get_all_configs()), 43)
        self.assertEqual(len(cr.validated_runhistory.data), 27)
        self.assertEqual(len(cr.validated_runhistory.get_all_configs()), 3)
        self.assertEqual(len(cr.combined_runhistory.data), 126)
        self.assertEqual(len(cr.combined_runhistory.get_all_configs()), 45)
