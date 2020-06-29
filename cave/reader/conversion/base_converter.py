import logging


class BaseConverter(object):
    """
    BaseConverter to inherit new converters from. This is the preferred method to create support for new file-formats.
    Please note, that you will need to implement the convert-method, which has to return a dictionary as specified.
    You can pass additional (arbitrary) python objects to CAVE by simply placing them in the returned dictionary.
    All custom key-value pairs in the dictionary will be available in CAVE's
    `RunsContainer <apidoc/cave.reader.runs_container>`_ as a dictionary `RunsContainer.share_information`.
    """
    def __init__(self):
        self.logger = logging.getLogger(self.__module__ + '.' + self.__class__.__name__)

    def convert(self, folders, ta_exec_dirs=None, output_dir=None, converted_dest='converted_input_data'):
        """Convert specific format results into SMAC3-format.

        Parameters
        ----------
        folders: List[str]
            list of parallel configurator-runs (folder paths!)
        ta_exec_dirs: List[str]
            only if you need to load instances, this is the path(s) from which the paths in the scenario are valid
        output_dir: str
            path to CAVE's output-directory
        converted_dest: str
            optional, this will be the parent folder in CAVE's output in which the converted runs (in SMAC-format) are
            saved, if not specified, will use temporary folders

        Returns
        -------
        result: dictionary
            .. code-block:: python

              dict{
                original_folder : dict{
                  'new_path' : converted_folder_path,
                  'config_space' : config_space,
                  'runhistory' : runhistory,
                  'validated_runhistory' : validated_runhistory,
                  'scenario' : scenario,
                  'trajectory' : trajectory,
                }
              }

            in addition, the result-dictionary can contain any number of arbitrary key-value pairs, that will be
            available in CAVE's `RunsContainer`
        """
        raise NotImplementedError()