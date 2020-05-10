from cave.analyzer.base_analyzer import BaseAnalyzer
from cave.feature_analysis.feature_analysis import FeatureAnalysis
from cave.utils.helpers import check_for_features
from cave.utils.hpbandster_helpers import format_budgets


class BoxViolin(BaseAnalyzer):
    """
    Box and Violin Plots show the distribution of each feature value across the instances.  Box plots show the
    quantiles of the distribution and violin plots show the approximated probability density of the feature values.
    Such plots are useful to inspect the instances and to detect characteristics of the instances. For example, if
    the distributions have two or more modes, it could indicate that the instance set is heterogeneous which could
    cause problems in combination with racing strategies configurators typically use. NaN values are removed from
    the data."""

    def __init__(self,
                 runscontainer,
                 ):
        super().__init__(runscontainer)
        check_for_features(runscontainer.scenario)

        formatted_budgets = format_budgets(self.runscontainer.get_budgets())
        for budget, run in zip(self.runscontainer.get_budgets(),
                               self.runscontainer.get_aggregated(keep_budgets=True, keep_folders=False)):
            self.result[formatted_budgets[budget]] = self.box_violin(
                output_dir=run.output_dir,
                scenario=run.scenario,
                feat_names=run.feature_names,
                feat_importance=run.share_information['feature_importance'],
            )

    def get_name(self):
        return "Violin and Box Plots"

    def box_violin(self,
                   output_dir,
                   scenario,
                   feat_names,
                   feat_importance,
                   ):
        feat_analysis = FeatureAnalysis(output_dn=output_dir,
                                        scenario=scenario,
                                        feat_names=feat_names,
                                        feat_importance=feat_importance)
        return {_[0] : {'figure' : _[1]} for _ in feat_analysis.get_box_violin_plots()}
