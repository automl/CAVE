import glob
import logging
import os
from contextlib import contextmanager

from cave.utils.exceptions import NotUniqueError


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

        self.scen = None

    def get_scenario(self):
        """Expects `self.folder/scenario.txt` with appropriately formatted
        scenario-information (`<https://automl.github.io/SMAC3/stable/options.html#scenario>`_)"""
        raise NotImplemented()

    def get_runhistory(self):
        """Create RunHistory-object from files."""
        raise NotImplemented()

    def get_validated_runhistory(self):
        """Create validated runhistory from files, if available."""
        raise NotImplemented()

    def get_trajectory(self):
        """Create trajectory (list with dicts as entries)"""
        raise NotImplemented()

    @classmethod
    def check_for_files(cls):
        raise NotImplemented()

    @classmethod
    def get_glob_file(cls, folder, fn, raise_on_failure=True):
        """
        If a file is not found in the expected path structure, we can check if it's unique in the subfolders and if so, return it.
        """
        globbed = glob.glob(os.path.join(folder, '**', fn), recursive=True)
        if len(globbed) == 1:
            return globbed[0]
        elif len(globbed) < 1:
            if raise_on_failure:
                raise FileNotFoundError("The file \"{}\" does not exist in \"{}\".".format(fn, folder))
        elif len(globbed) > 1:
            if raise_on_failure:
                raise NotUniqueError("The file \"{}\" exists {} times in \"{}\", but not in the expected place.".format(
                    fn, len(globbed), folder))
        return ""