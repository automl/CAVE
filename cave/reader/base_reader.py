import os
from contextlib import contextmanager
import logging

@contextmanager
def changedir(newdir):
    olddir = os.getcwd()
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(olddir)

class BaseReader(object):
    """Abstract base class to inherit reader from. Reader load necessary objects
    (scenario, runhistory, trajectory) from files for different formats."""

    def __init__(self, folder, ta_exec_dir):
        self.logger = logging.getLogger("cave.reader")
        self.folder = folder
        self.ta_exec_dir = ta_exec_dir

    def get_scenario(self):
        """Create Scenario-object from files."""
        raise NotImplemented()

    def get_runhistory(self):
        """Create RunHistory-object. Returns (original_runhistory,
        validated_runhistory) where validated_runhistory can be None."""
        raise NotImplemented()

    def get_trajectory(self):
        """Create trajectory (list with dicts as entries)"""
        raise NotImplemented()