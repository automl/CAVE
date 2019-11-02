import os
from collections import OrderedDict

from cave.analyzer.cave_parameter_importance import CaveParameterImportance


class CaveAblation(CaveParameterImportance):
    """ Ablation Analysis is a method to determine parameter importance by comparing two parameter configurations,
    typically the default and the optimized configuration.  It uses a greedy forward search to determine the order
    of flipping the parameter settings from default configuration to incumbent such that in each step the cost is
    maximally decreased."""

    def __init__(self,
                 runscontainer,
                 marginal_threshold=0.05):
        super().__init__(runscontainer)

        self.marginal_threshold = marginal_threshold

        self.parameter_importance("ablation")

    def get_name(self):
        return "Ablation"

    def postprocess(self, pimp, output_dir):
        result = OrderedDict([
            ('figure', [os.path.join(output_dir, fn) for fn in ["ablationpercentage.png", "ablationperformance.png"]])
        ])
        return result

