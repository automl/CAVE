from collections import OrderedDict

from pandas import DataFrame

from cave.analyzer.apt.base_apt import BaseAPT


class APTOverview(BaseAPT):
    """
    Overview of AutoPyTorch-Specific Configurations
    """
    def __init__(self, runscontainer):
        super().__init__(runscontainer)
        self.output_dir = runscontainer.output_dir

        html_table = self.run()
        self.result["General"] = {"table": html_table,
                                  "tooltip": "AutoPyTorch configuration."}

    def get_name(self):
        return "APT Overview"

    def run(self):
        """ Generate tables. """
        config_dict = self.runscontainer.autonet["autonet"].get_current_autonet_config()

        html_table = DataFrame(data=OrderedDict([('Autonet Configuration', config_dict)]))
        html_table = html_table.reindex(list(config_dict.keys()))
        html_table = html_table.to_html(escape=False, header=False, justify='left')

        return html_table