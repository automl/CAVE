from collections import OrderedDict

from pimp.importance.importance import Importance
from ConfigSpace.configuration_space import Configuration

from cave.analyzer.base_analyzer import BaseAnalyzer
from cave.utils.hpbandster_helpers import format_budgets


class CaveParameterImportance(BaseAnalyzer):

    def __init__(self,
                 runscontainer,
                 ):
        """Calculate parameter-importance using the PIMP-package.
        """
        super().__init__(runscontainer)


    def parameter_importance(self, modus):
        """
        modus: str
            modus for parameter importance, from
            [forward-selection, ablation, fanova, lpi]
        """
        runs_by_budget = self.runscontainer.get_aggregated(True, False)

        formatted_budgets = format_budgets(self.runscontainer.get_budgets(), allow_whitespace=True)

        for run in runs_by_budget:
            self.logger.info("... parameter importance {} on {}".format(modus, run.get_identifier()))
            if not formatted_budgets[run.budget] in self.result:
                self.result[formatted_budgets[run.budget]] = OrderedDict()
            try:
                run.pimp.evaluate_scenario([modus], run.output_dir)
            except RuntimeError as e:
                err = "Encountered error '{}' for '{}' in '{}', (for fANOVA this can e.g. happen with too few data-points).".format(
                                e, run.get_identifier(), modus)
                self.logger.info(err, exc_info=1)
                self.result[formatted_budgets[run.budget]][modus + '_error'] = err
                continue
            individual_result = self.postprocess(run.pimp, run.output_dir)
            self.result[formatted_budgets[run.budget]] = individual_result

            run.share_information['parameter_importance'][modus] = run.pimp.evaluator.evaluated_parameter_importance
            run.share_information['evaluators'][modus] = run.pimp.evaluator

