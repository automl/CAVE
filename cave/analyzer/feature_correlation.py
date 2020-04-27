from cave.analyzer.base_analyzer import BaseAnalyzer
from cave.feature_analysis.feature_analysis import FeatureAnalysis
from cave.utils.helpers import check_for_features
from cave.utils.hpbandster_helpers import format_budgets


class FeatureCorrelation(BaseAnalyzer):
    """
    Correlation of features based on the Pearson product-moment correlation. Since instance features are used to train an
    empirical performance model in model-based configurators, it can be important to remove correlated features in a
    pre-processing step depending on the machine-learning algorithm.  Darker fields corresponds to a larger correlation
    between the features."""

    def __init__(self,
                 runscontainer,
                 ):
        super().__init__(runscontainer)
        check_for_features(runscontainer.scenario)

        formatted_budgets = format_budgets(self.runscontainer.get_budgets())
        for budget, run in zip(self.runscontainer.get_budgets(),
                               self.runscontainer.get_aggregated(keep_budgets=True, keep_folders=False)):
            self.result[formatted_budgets[budget]] = self.feat_analysis(
                output_dir=run.output_dir,
                scenario=run.scenario,
                feat_names=run.feature_names,
                feat_importance=run.share_information['feature_importance'],
            )

    def get_name(self):
        return "Feature Correlation"

    def feat_analysis(self,
                      output_dir,
                      scenario,
                      feat_names,
                      feat_importance,
                      ):
        feat_analysis = FeatureAnalysis(output_dn=output_dir,
                                        scenario=scenario,
                                        feat_names=feat_names,
                                        feat_importance=feat_importance)

        ##  feat_analysis.correlation_plot()  # Generate an additional plot
        return {'figure' :  feat_analysis.correlation_plot(imp=False)}
