import os

from tensorboard import program
from collections import OrderedDict

from pandas import DataFrame

from cave.analyzer.apt.base_apt import BaseAPT
from cave.utils.exceptions import NotApplicable
from cave.utils.hpbandster_helpers import get_incumbent_trajectory


class APTTensorboard(BaseAPT):
    """
    Overview of AutoPyTorch-Specific Configurations
    """
    def __init__(self, runscontainer):
        super().__init__(runscontainer)
        self.result = {}
        self.run()

    def get_name(self):
        return "APT Tensorboard"

    def run(self):
        apt_basedir = os.path.join(self.runscontainer.output_dir, "apt_tensorboard")
        #self.runscontainer.autopytorch["autopytorch"].update_autonet_config(result_logger_dir=apt_basedir)

        # APT produces one BOHB-result as of now. If this changes, we will need to aggregate the trajectory over
        # multiple BOHB-results as in cost_over_time.
        if len(self.runscontainer.get_bohb_results()) != 1:
            self.logger.error("AutoPyTorch is not expected to produce multiple BOHB/HyperBand-results as of now.")
            raise NotApplicable("AutoPyTorch is not expected to produce multiple BOHB/HyperBand-results as of now.")
        traj = get_incumbent_trajectory(self.runscontainer.get_bohb_results()[0], mode='racing')

        for config in traj['config']:
            self.runscontainer.get_tensorboard_result(config['config'])

        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', apt_basedir])
        url = tb.launch()
        print(url)
        self.result[None] = " <iframe src=" + url + "></iframe> "