import os
import shutil
import tempfile

from cave.reader.conversion.base_converter import BaseConverter
from cave.reader.conversion.hpbandster2smac import HpBandSter2SMAC


class APT2SMAC(BaseConverter):

    def convert(self, folders, ta_exec_dirs=None, output_dir=None, converted_dest='converted_input_data'):
        self.logger.debug(
            "Converting APT-data to SMAC3-data. Called with: folders=%s, ta_exec_dirs=%s, output_dir=%s, "
            "converted_dest=%s", str(folders), str(ta_exec_dirs), str(output_dir), str(converted_dest))

        # Using temporary files for the intermediate smac-result-like format if no output_dir specified
        if not output_dir:
            output_dir = tempfile.mkdtemp()
            self.logger.debug("Temporary directory for intermediate SMAC3-results: %s", output_dir)
        if ta_exec_dirs is None or len(ta_exec_dirs) == 0:
            ta_exec_dirs = ['.']
        if len(ta_exec_dirs) != len(folders):
            ta_exec_dirs = [ta_exec_dirs[0] for _ in folders]

        self.logger.info("Assuming APT builds on hpbandster-format...")
        results = HpBandSter2SMAC().convert(folders, ta_exec_dirs, output_dir, converted_dest)

        self.logger.info("Assuming APT logs in tensorboard-files")
        tf_paths = {}
        for folder, result in results.items():
            self.logger.debug("Checking for tensorflow-event files in %s", folder)
            tf_paths[folder] = []
            for filename in os.listdir(folder):
                if 'tfevents' in filename:
                    dst = shutil.copyfile(filename, os.path.join(result['new_path'], filename))
                    tf_paths[folder].append(dst)
        for f, paths in tf_paths.items():
            if len(paths) == 0:
                self.logger.warning("No tfevents-files found for APT-folder %s!", f)
            results[f]['tfevents_paths'] = paths

        return results