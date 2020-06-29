import os

from cave.analyzer.base_analyzer import BaseAnalyzer
from cave.utils.apt_helpers.apt_warning import apt_warning
from cave.utils.exceptions import Deactivated


class APTTensorboard(BaseAnalyzer):
    """
    Overview of AutoPyTorch-Specific Configurations
    """
    def __init__(self, runscontainer):
        super().__init__(runscontainer)
        if self.runscontainer.file_format != "APT":
            raise Deactivated("{} deactivated, only designed for file-format APT (but detected {})".format(
                self.get_name(), self.runscontainer.file_format
            ))
        apt_warning(self.logger)
        self.run()

    def get_name(self):
        return "Auto-PyTorch Tensorboard"

    def run(self):
        try:
            from tensorboard import program
        except ModuleNotFoundError:
            raise Deactivated("Please install tensorboard to perform this analysis!")

        if len(self.runscontainer.get_folders()) != 1:
            raise ValueError("Undefined behaviour for multiple APT-outputs...")
        run = self.runscontainer.get_aggregated(keep_budgets=False, keep_folders=True)[0]

        # This line will need to be adapted
        single_tfevents_file = run.share_information['tfevents_paths'][0]
        tfevents_dir = os.path.split(single_tfevents_file)[0]
        self.logger.info("Tensorboard base dir: %s", tfevents_dir)
        print(tfevents_dir)

        #for config in traj['config']:
        #    self.runscontainer.get_tensorboard_result(config['config'])

        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', tfevents_dir])
        url = tb.launch()

        self.result["else"] = " <iframe src=" + url + " width=\"950\" height=\"700\"></iframe> "