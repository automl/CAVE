import logging
import os


class BaseConverter(object):

    def __init__(self):
        self.logger = logging.getLogger(self.__module__ + '.' + self.__class__.__name__)

    def convert(self, folders, ta_exec_dirs=None, output_dir=None, converted_dest='converted_input_data'):
        """Convert specific format results into SMAC-format.

        Parameters
        ----------
        folders: List[str]
            list of parallel configurator-runs (folder paths!)
        ta_exec_dirs: List[str]
            only relevant if you need to load instances, this should be the path from which the paths in the scenario are valid
        output_dir: str
            path to CAVE's output-directory
        converted_dest: str
            optional, this will be the parent folder in the output in which the converted runs (in SMAC-format) are saved
            if not specified, will use temporary folders

        Returns
        -------
        result: dict{f : dict{
                'new_path' : converted_folder_path,
                'config_space' : config_space,
                'runhistory' : runhistory,
                'validated_runhistory' : validated_runhistory,
                'scenario' : scenario,
                'trajectory' : trajectory}}
        """
        raise NotImplementedError()

    def get_folder_basenames(self, folders):
        """Shorten folder-strings as much as possible (always keeping the basename).
        ["foo/bar/run_1", "foo/bar/run_2/"] will be ["run_1", "run_2]
        ["foo/run_1/bar/", "foo/run_2/bar"] will be ["run_1/bar", "run_2/bar"]
        """
        throw, keep = folders[:], ['' for _ in range(len(set(folders)))]
        max_parts = max([len(f.split('/')) for f in folders])
        for _ in range(max_parts):
            for idx in range(len(folders)):
                throw[idx], new = os.path.split(throw[idx].rstrip('/'))
                keep[idx] = os.path.join(new, keep[idx]).rstrip('/')
            if len(set(keep)) == len(set(folders)):
                break

        return keep