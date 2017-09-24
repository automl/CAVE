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
                    "--param_importance", "False"]
        with mock.patch.object(sys, 'argv', testargs):
            self.cli.main_cli()

    def test_call_from_local_path(self):
        with changedir("examples/spear_qcp_small"):
            testargs = ["python", "scripts/spy.py", "--folders",
                        "example_output_1", "--verbose", "DEBUG",
                        "--param_importance", "False"]
            with mock.patch.object(sys, 'argv', testargs):
                self.cli.main_cli()

    def test_param_importance(self):
        testargs = ["python", "scripts/spy.py", "--folders",
                    "examples/spear_qcp_small/example_output_1",
                    "--verbose", "DEBUG",
                    "--ta_exec_dir", "examples/spear_qcp_small",
                    "--param_importance", "true"]
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
                    "--param_importance", "False"]
        with mock.patch.object(sys, 'argv', testargs):
            self.cli.main_cli()

    def test_notrain_notest_nofeat(self):
        """
        Testing analysis w/o train- or test-insts (and without features)
        Optimized on quality, deterministic, without instance-specifics.
        """
        with changedir("test/test_files/branin"):
            testargs = ["python", "../../../scripts/spy.py", "--folders",
                        "qual_notrain_notest_nofeat_run1",
                        "qual_notrain_notest_nofeat_run2",
                        "--verbose", "DEBUG",
                        "--output", "qual_notrain_notest_nofeat_SPY",
                        "--param_importance", "False"]
            with mock.patch.object(sys, 'argv', testargs):
                self.cli.main_cli()

    def test_train_notest_nofeat(self):
        """
        Testing analysis with train- but without test-insts (and without features)
        Optimized on quality, deterministic, with and without instance-specifics.
        """
        with changedir("test/test_files/branin"):
            testargs = ["python", "../../../scripts/spy.py", "--folders",
                        "qual_train_nospecs_notest_nofeat_run1",
                        "--verbose", "DEBUG",
                        "--output", "qual_train_nospecs_notest_nofeat_SPY",
                        "--param_importance", "False"]
            with mock.patch.object(sys, 'argv', testargs):
                self.cli.main_cli()
            # With instance specifics
            testargs[3] = "qual_train_specs_notest_nofeat_run1"
            testargs[7] = "qual_train_specs_notest_nofeat_SPY"
            with mock.patch.object(sys, 'argv', testargs):
                self.cli.main_cli()

    def test_train_test_nofeat(self):
        """
        Testing analysis with train- and test-insts (and without features)
        Optimized on quality, deterministic, with and without instance-specifics.
        """
        with changedir("test/test_files/branin"):
            testargs = ["python", "../../../scripts/spy.py", "--folders",
                        "qual_train_nospecs_test_nofeat_run1",
                        "--verbose", "DEBUG",
                        "--output", "qual_train_nospecs_test_nofeat_SPY",
                        "--param_importance", "False"]
            with mock.patch.object(sys, 'argv', testargs):
                self.cli.main_cli()
            # With instance specifics
            testargs[3] = "qual_train_specs_test_nofeat_run1"
            testargs[7] = "qual_train_specs_test_nofeat_SPY"
            with mock.patch.object(sys, 'argv', testargs):
                self.cli.main_cli()

    def test_train_test_feat(self):
        """
        Testing analysis with train- and test-insts (and with features)
        Optimized on quality, deterministic, with and without instance-specifics.
        """
        with changedir("test/test_files/branin"):
            testargs = ["python", "../../../scripts/spy.py", "--folders",
                        "qual_train_nospecs_test_feat_run1",
                        "--verbose", "DEBUG",
                        "--output", "qual_train_nospecs_test_feat_SPY",
                        "--param_importance", "False"]
            with mock.patch.object(sys, 'argv', testargs):
                self.cli.main_cli()
            # With instance specifics
            testargs[3] = "qual_train_specs_test_feat_run1"
            testargs[7] = "qual_train_specs_test_feat_SPY"
            with mock.patch.object(sys, 'argv', testargs):
                self.cli.main_cli()

    # TODO tests on runtime
