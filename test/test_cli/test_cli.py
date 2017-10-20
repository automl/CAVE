import sys
import os
import numpy as np
import logging
import shutil
from contextlib import contextmanager

import unittest
from unittest import mock

from spysmac.spysmac_cli import SpySMACCLI
from spysmac.spyfacade import SpySMAC

@contextmanager
def changedir(newdir):
    olddir = os.getcwd()
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(olddir)

class TestCLI(unittest.TestCase):

    def setUp(self):
        self.cli = SpySMACCLI()
        self.output = "test/test_files/cli_test_output"
        shutil.rmtree(self.output, ignore_errors=True)

        self.testargs = ["python", "scripts/spy.py", "--folders",
                         "examples/spear_qcp_small/example_output_1",
                         "--verbose", "DEBUG",
                         "--ta_exec_dir", "examples/spear_qcp_small",
                         "--param_importance", "none",
                         "--feat_analysis", "none"]

    @mock.patch('spysmac.spyfacade.SpySMAC.complete_data')
    @mock.patch('spysmac.spyfacade.SpySMAC.analyze')
    def test_call_from_local_path(self, mock_ana, mock_data):
        with changedir("examples/spear_qcp_small"):
            testargs = self.testargs
            testargs.remove('--ta_exec_dir')
            testargs.remove('examples/spear_qcp_small')
            i = testargs.index('examples/spear_qcp_small/example_output_1')
            del testargs[i]
            testargs.insert(i, 'example_output_1')
            testargs.extend(['--validation', 'epm'])

            with mock.patch.object(sys, 'argv', testargs):
                self.cli.main_cli()

    @mock.patch('spysmac.spyfacade.SpySMAC.complete_data')
    @mock.patch('spysmac.spyfacade.SpySMAC.analyze')
    def test_feature_analysis(self, mock_ana, mock_data):
        """ Testing whether feature_analysis cmdline-args are processed
        correctly. """
        testargs = self.testargs
        feat_index = testargs.index('--feat_analysis') + 1

        # Test 'all' feature analysis
        testargs[feat_index] = 'all'
        with mock.patch.object(sys, 'argv', testargs):
            self.cli.main_cli()
            mock_ana.assert_called_once_with(
                param_importance=[],
                cdf=True, confviz=True, performance=True, scatter=True,
                feature_analysis=['box_violin', 'correlation', 'feat_importance',
                                  'clustering', 'feature_cdf'])

        # Test 'none' feature analysis
        testargs[feat_index] = 'none'
        with mock.patch.object(sys, 'argv', testargs):
            self.cli.main_cli()
            mock_ana.assert_called_with(
                param_importance=[],
                cdf=True, confviz=True, performance=True, scatter=True,
                feature_analysis=[])

        # test combination 'box_violin' and 'correlation'
        testargs[feat_index] = 'box_violin'
        testargs.insert(feat_index+1, 'correlation')
        with mock.patch.object(sys, 'argv', testargs):
            self.cli.main_cli()
            mock_ana.assert_called_with(
                param_importance=[],
                cdf=True, confviz=True, performance=True, scatter=True,
                feature_analysis=['box_violin', 'correlation'])

    @mock.patch('spysmac.spyfacade.SpySMAC.complete_data')
    @mock.patch('spysmac.spyfacade.SpySMAC.analyze')
    def test_param_importance(self, mock_ana, mock_data):
        """ Testing whether param_importance cmdline-args are processed
        correctly. """
        testargs = self.testargs
        param_index = testargs.index('--param_importance') + 1

        # Test 'all' feature analysis
        testargs[param_index] = 'all'
        with mock.patch.object(sys, 'argv', testargs):
            self.cli.main_cli()
            mock_ana.assert_called_once_with(
                cdf=True, confviz=True, performance=True, scatter=True,
                feature_analysis=[], param_importance=['ablation',
                    'forward_selection', 'fanova'])

        # Test 'none' feature analysis
        testargs[param_index] = 'none'
        with mock.patch.object(sys, 'argv', testargs):
            self.cli.main_cli()
            mock_ana.assert_called_with(
                cdf=True, confviz=True, performance=True, scatter=True,
                feature_analysis=[], param_importance=[])

        # test combination 'ablation' and'fanova'
        testargs[param_index] = 'ablation'
        testargs.insert(param_index+1, 'fanova')
        with mock.patch.object(sys, 'argv', testargs):
            self.cli.main_cli()
            mock_ana.assert_called_with(
                cdf=True, confviz=True, performance=True, scatter=True,
                feature_analysis=[], param_importance=['ablation', 'fanova'])

    @mock.patch('spysmac.spyfacade.SpySMAC.complete_data')
    @mock.patch('spysmac.spyfacade.SpySMAC.analyze')
    def test_run_with_multiple_folders(self, mock_ana, mock_data):
        """
        Testing basic functionality.
        """
        # Run for 5 algo-calls
        testargs = self.testargs
        folder_index = testargs.index('--folders') + 1
        testargs.insert(folder_index,
                "examples/spear_qcp_small/example_output_2")
        testargs.insert(folder_index,
                "examples/spear_qcp_small/example_output_3")
        with mock.patch.object(sys, 'argv', testargs):
            self.cli.main_cli()

    #TODO this is not really a unittest, but a cornercasing
    def test_branin_corners(self):
        """ Testing all possible combinations, using the branin target
        algorithm. """

        def run(case):
            """
            Testing analysis w/o train- or test-insts (and without features)
            Optimized on quality, deterministic, without instance-specifics.
            """
            print("running: {}".format(case))
            with changedir("test/test_files/branin"):
                testargs = ["python", "../../../scripts/spy.py", "--folders",
                            case+"_run1",
                            "--verbose", "DEBUG",
                            "--feat_analysis", "none",
                            "--output", case+"_SPY",
                            "--param_importance", "none"]
                with mock.patch.object(sys, 'argv', testargs):
                    self.cli.main_cli()

        for case in ["qual_notrain_notest_nofeat",
                     "qual_train_nospecs_notest_nofeat",
                     "qual_train_specs_notest_nofeat",
                     "qual_train_nospecs_test_nofeat",
                     "qual_train_specs_test_nofeat",
                     "qual_train_nospecs_test_feat",
                     "qual_train_specs_test_feat"]:
            run(case)
