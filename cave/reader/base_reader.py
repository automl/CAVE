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

    def __init__(self, folder, ta_exec_dir):
        self.logger = logging.getLogger("cave.reader")
        self.folder = folder
        self.ta_exec_dir = ta_exec_dir

    def get_scenario(self):
        raise NotImplemented()

    def get_runhistory(self):
        raise NotImplemented()

    def get_trajectory(self):
        raise NotImplemented()
