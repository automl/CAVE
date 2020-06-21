import os

from pandas import DataFrame

from cave.analyzer.base_analyzer import BaseAnalyzer
from cave.utils.helpers import check_for_features
from cave.utils.hpbandster_helpers import format_budgets


class FeatureImportance(BaseAnalyzer):
    def __init__(self,
                 runscontainer,
                 ):
        super().__init__(runscontainer)
        check_for_features(runscontainer.scenario)

        formatted_budgets = format_budgets(self.runscontainer.get_budgets())
        for budget, run in zip(self.runscontainer.get_budgets(),
                               self.runscontainer.get_aggregated(keep_budgets=True, keep_folders=False)):
            feat_imp, plots = self.feature_importance(
                pimp=run.pimp,
                output_dir=run.output_dir,
            )
            self.result[formatted_budgets[budget]] = plots
           # Add to run so other analysis-methods can use the information
            run.share_information['feature_importance'] = feat_imp

    def get_name(self):
        return "Feature Importance"

    def feature_importance(self, pimp, output_dir):
        self.logger.info("... plotting feature importance")

        old_values = (pimp.forwardsel_feat_imp, pimp._parameters_to_evaluate, pimp.forwardsel_cv)
        pimp.forwardsel_feat_imp = True
        pimp._parameters_to_evaluate = -1
        pimp.forwardsel_cv = False

        dir_ = os.path.join(output_dir, 'feature_plots/importance')
        os.makedirs(dir_, exist_ok=True)
        res = pimp.evaluate_scenario(['forward-selection'], dir_)
        feat_importance = res[0]['forward-selection']['imp']

        plots = [os.path.join(dir_, 'forward-selection-barplot.png'),
                 os.path.join(dir_, 'forward-selection-chng.png')]
        # Restore values
        pimp.forwardsel_feat_imp, pimp._parameters_to_evaluate, pimp.forwardsel_cv = old_values

        table = DataFrame(data=list(feat_importance.values()), index=list(feat_importance.keys()), columns=["Error"])
        table = table.to_html()

        result = {'Table': {'table': table}}
        for p in plots:
            result[os.path.splitext(os.path.basename(p))[0]] = {'figure' : p}
        return (feat_importance, result)
