import sys
import os
import numpy as np
import logging
import shutil
from contextlib import contextmanager

import unittest
from unittest import mock

from spysmac.spysmac_cli import SpySMACCLI

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

    def test_no_output(self):
        testargs = ["python", "scripts/spy.py", "--folders",
                    "examples/spear_qcp_small/example_output_1",
                    "--verbose", "DEBUG",
                    "--ta_exec_dir", "examples/spear_qcp_small",
                    "--param_importance", "none",
                    "--feat_analysis", "none"
                    ]
        with mock.patch.object(sys, 'argv', testargs):
            self.cli.main_cli()

    def test_call_from_local_path(self):
        with changedir("examples/spear_qcp_small"):
            testargs = ["python", "scripts/spy.py", "--folders",
                        "example_output_1", "--verbose", "DEBUG",
                        "--feat_analysis", "none",
                        "--param_importance", "none"]
            with mock.patch.object(sys, 'argv', testargs):
                self.cli.main_cli()

    def test_feature_analysis(self):
        testargs = ["python", "scripts/spy.py", "--folders",
                    "examples/spear_qcp_small/example_output_1",
                    "--verbose", "DEBUG",
                    "--ta_exec_dir", "examples/spear_qcp_small",
                    "--feat_analysis", "all",
                    "--param_importance", "none"]
        with mock.patch.object(sys, 'argv', testargs):
            self.cli.main_cli()

    def test_param_importance(self):
        testargs = ["python", "scripts/spy.py", "--folders",
                    "examples/spear_qcp_small/example_output_1",
                    "--verbose", "DEBUG",
                    "--ta_exec_dir", "examples/spear_qcp_small",
                    "--feat_analysis", "none",
                    "--param_importance", "all"]
        with mock.patch.object(sys, 'argv', testargs):
            self.cli.main_cli()

    def test_run_with_multiple_folders(self):
        """
        Testing basic functionality.
        """
        # Run for 5 algo-calls
        testargs = ["python", "scripts/spy.py", "--folders",
                    "examples/spear_qcp_small/example_output_1",
                    "examples/spear_qcp_small/example_output_2",
                    "examples/spear_qcp_small/example_output_3",
                    "--verbose", "DEBUG", "--output", self.output,
                    "--ta_exec_dir", "examples/spear_qcp_small",
                    "--missing_data_method", "epm",
                    "--feat_analysis", "none",
                    "--param_importance", "none"]
        with mock.patch.object(sys, 'argv', testargs):
            self.cli.main_cli()

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
