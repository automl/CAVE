from pandas import DataFrame

from cave.analyzer.base_analyzer import BaseAnalyzer


class CompareDefaultIncumbent(BaseAnalyzer):
    """
    Comparing parameters of default and incumbent. Parameters that differ from default to incumbent are presented
    first. Parameters that are inactive for both configurations are omitted.
    """

    def __init__(self, runscontainer):
        """ Create comparison table of default and incumbent
        Removes unused parameters.
        """
        super().__init__(runscontainer)

        default = self.runscontainer.scenario.cs.get_default_configuration()
        runs = self.runscontainer.get_runs_for_budget(self.runscontainer.get_highest_budget())
        incumbents = {r.incumbent : r.epm_runhistory.get_cost(r.incumbent) for r in runs}
        incumbent = max(incumbents, key=lambda key: incumbents[key])

        # Remove unused parameters
        keys = [k for k in default.configuration_space.get_hyperparameter_names() if default[k] or incumbent[k]]
        default = [default[k] if default[k] is not None else "inactive" for k in keys]
        incumbent = [incumbent[k] if incumbent[k] is not None else "inactive" for k in keys]
        zipped = list(zip(keys, default, incumbent))
        # Show first parameters that changed
        same = [x for x in zipped if x[1] == x[2]]
        diff = [x for x in zipped if x[1] != x[2]]
        table = []
        if len(diff) > 0:
            table.extend([(15 * '-' + ' Changed parameters: ' + 15 * '-', 5 * '-', 5 * '-')])
            table.extend(diff)
        if len(same) > 0:
            table.extend([(15 * '-' + ' Unchanged parameters: ' + 15 * '-', 5 * '-', 5 * '-')])
            table.extend(same)
        keys, table = [k[0] for k in table], [k[1:] for k in table]
        df = DataFrame(data=table, columns=["Default", "Incumbent"], index=keys)
        self.result['table'] = df.to_html()

    def get_name(self):
        return "Best Configuration"
