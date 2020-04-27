from cave.analyzer.base_analyzer import BaseAnalyzer
from cave.feature_analysis.feature_analysis import FeatureAnalysis
from cave.utils.helpers import check_for_features
from cave.utils.hpbandster_helpers import format_budgets


class FeatureClustering(BaseAnalyzer):
    """ Clustering instances in 2d; the color encodes the cluster assigned to each cluster. Similar to ISAC, we use
    a k-means to cluster the instances in the feature space. As pre-processing, we use standard scaling and a PCA to
    2 dimensions. To guess the number of clusters, we use the silhouette score on the range of 2 to 12 in the number
    of clusters"""

    def __init__(self, runscontainer):
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
        return "Feature Clustering"

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

        return {'figure' : feat_analysis.cluster_instances()}

