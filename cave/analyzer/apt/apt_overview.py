import os
from collections import OrderedDict

from pandas import DataFrame

from cave.analyzer.base_analyzer import BaseAnalyzer
from cave.utils.apt_helpers.apt_warning import apt_warning
from cave.utils.exceptions import Deactivated


class APTOverview(BaseAnalyzer):
    """
    Overview of AutoPyTorch-Specific Configurations
    """
    def __init__(self, runscontainer):
        super().__init__(runscontainer)
        self.output_dir = runscontainer.output_dir

        if self.runscontainer.file_format != "APT":
            raise Deactivated("{} deactivated, only designed for file-format APT (but detected {})".format(
                self.get_name(), self.runscontainer.file_format
            ))

        apt_warning(self.logger)

        html_table = self.run()
        self.result["General"] = {"table": html_table,
                                  "tooltip": "AutoPyTorch configuration."}

    def get_name(self):
        return "Auto-PyTorch Overview"

    def run(self):
        """ Generate tables. """
        # Run-specific / budget specific infos
        runs = self.runscontainer.get_aggregated(keep_folders=True, keep_budgets=False)
        apt_config_dict = self._runspec_dict_apt_config(runs)
        results_fit_dict = self._runspec_dict_results_fit(runs)

        for k, runspec_dict in [("Auto-PyTorch Configuration", apt_config_dict),
                                ("Results of the fit()-call", results_fit_dict)]:
            order_spec = list(list(runspec_dict.values())[0].keys())  # Get keys of any sub-dict for order
            html_table_specific = DataFrame(runspec_dict)
            html_table_specific = html_table_specific.reindex(order_spec)
            html_table_specific = html_table_specific.to_html(escape=False, justify='left')

            self.result[k] = {"table": html_table_specific}

    def _runspec_dict_results_fit(self, runs):
        runspec = OrderedDict()

        for idx, run in enumerate(runs):
            self.logger.debug("Path to folder for run no. {}: {}".format(idx, str(run.path_to_folder)))
            name = os.path.basename(run.path_to_folder)
            runspec[name] = OrderedDict()
            for k, v in run.share_information['results_fit']['info'].items():
                runspec[name]["Info: " + str(k)] = v
            for k, v in run.share_information['results_fit']['optimized_hyperparameter_config'].items():
                runspec[name]["Parameter: " + str(k)] = v
            runspec[name]["Budget"] = run.share_information['results_fit']['budget']
            runspec[name]["Loss"] = run.share_information['results_fit']['loss']

        return runspec

    def _runspec_dict_apt_config(self, runs):
        runspec = OrderedDict()

        for idx, run in enumerate(runs):
            self.logger.debug("Path to folder for run no. {}: {}".format(idx, str(run.path_to_folder)))
            name = os.path.basename(run.path_to_folder)
            runspec[name] = OrderedDict()
            for k, v in run.share_information['apt_config'].items():
                runspec[name][k] = v

        return runspec