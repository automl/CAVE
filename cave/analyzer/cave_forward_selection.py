import os
from collections import OrderedDict

from cave.analyzer.cave_parameter_importance import CaveParameterImportance


class CaveForwardSelection(CaveParameterImportance):
    """
    Forward Selection is a generic method to obtain a subset of parameters to achieve the same prediction error as
    with the full parameter set.  Each parameter is scored by how much the out-of-bag-error of an empirical
    performance model based on a random forest is decreased.
    """

    def __init__(self,
                 runscontainer,
                 marginal_threshold=0.05):
        super().__init__(runscontainer)

        self.marginal_threshold = marginal_threshold

        self.parameter_importance("forward-selection")

    def get_name(self):
        return "Forward Selection"

    def postprocess(self, pimp, output_dir):
        return OrderedDict([
            ('figure', [os.path.join(output_dir, fn) for fn in ["forward-selection-barplot.png", "forward-selection-chng.png"]])
        ])

