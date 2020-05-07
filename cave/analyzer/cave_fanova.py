import operator
import os
from collections import OrderedDict

from pandas import DataFrame

from cave.analyzer.cave_parameter_importance import CaveParameterImportance


class CaveFanova(CaveParameterImportance):
    """
    fANOVA (functional analysis of variance) computes the fraction of the variance in the cost space explained by
    changing a parameter by marginalizing over all other parameters, for each parameter (or for pairs of
    parameters). Parameters with high importance scores will have a large impact on the performance.  To this end, a
    random forest is trained as an empirical performance model on the available empirical data from the available
    runhistories.
    """

    def __init__(self,
                 runscontainer,
                 marginal_threshold=0.05):
        """Wrapper for parameter_importance to save the importance-object/ extract the results. We want to show the
        top X most important parameter-fanova-plots.

        Parameters
        ----------
        runscontainer: RunsContainer
            contains all important information about the configurator runs
        marginal_threshold: float
            parameter/s must be at least this important to be mentioned
        """
        super().__init__(runscontainer)

        self.marginal_threshold = marginal_threshold
        self.parameter_importance("fanova")

    def get_name(self):
        return 'fANOVA'

    def postprocess(self, pimp, output_dir):
        result = OrderedDict()

        def parse_pairwise(p):
            """parse pimp's way of having pairwise parameters as key as str and return list of individuals"""
            res = [tmp.strip('\' ') for tmp in p.strip('[]').split(',')]
            return res

        parameter_imp = {k: v * 100 for k, v in pimp.evaluator.evaluated_parameter_importance.items()}
        parameter_imp_std = {}
        if hasattr(pimp.evaluator, 'evaluated_parameter_importance_uncertainty'):
            parameter_imp_std = {k: v * 100 for k, v in pimp.evaluator.evaluated_parameter_importance_uncertainty.items()}

        for k in parameter_imp.keys():
            self.logger.debug("fanova-importance for %s: mean (over trees): %f, std: %s", k, parameter_imp[k],
                              str(parameter_imp_std[k]) if parameter_imp_std else 'N/A')

        # Split single and pairwise (pairwise are string: "['p1','p2']")
        fanova_single_imp = {k : v for k, v in parameter_imp.items() if not k.startswith('[') and v > self.marginal_threshold}
        pairwise_imp = {k : v for k, v in parameter_imp.items() if k.startswith('[') and v > self.marginal_threshold}

        # Set internal parameter importance for further analysis (such as parallel coordinates)
        self.fanova_single_importance = fanova_single_imp
        self.fanova_pairwise_importance = fanova_single_imp

        # Dicts to lists of tuples, sorted descending after importance
        fanova_single_imp = OrderedDict(sorted(fanova_single_imp.items(), key=operator.itemgetter(1), reverse=True))
        pairwise_imp = OrderedDict(sorted(pairwise_imp.items(), key=operator.itemgetter(1), reverse=True))

        # Create table
        table = []
        if len(fanova_single_imp) > 0:
            table.extend([(20*"-"+" Single importance: "+20*"-", 20*"-")])
            for k, v in fanova_single_imp.items():
                value = str(round(v, 4))
                if parameter_imp_std:
                    value += " +/- " + str(round(parameter_imp_std[k], 4))
                table.append((k, value))
        if len(pairwise_imp) > 0:
            table.extend([(20*"-"+" Pairwise importance: "+20*"-", 20*"-")])
            for k, v in pairwise_imp.items():
                name = ' & '.join(parse_pairwise(k))
                value = str(round(v, 4))
                if parameter_imp_std:
                    value += " +/- " + str(round(parameter_imp_std[k], 4))
                table.append((name, value))

        keys, fanova_table = [k[0] for k in table], [k[1:] for k in table]
        df = DataFrame(data=fanova_table, index=keys)
        result['Importance'] = {'table' : df.to_html(escape=False, header=False, index=True, justify='left')}

        # Get plot-paths
        result['Marginals'] = {p : {'figure' : os.path.join(output_dir, "fanova", p + '.png')} for p in fanova_single_imp.keys()}
        # Right now no way to access paths of the plots -> file issue
        pairwise_plots = {" & ".join(parse_pairwise(p)) : os.path.join(output_dir, 'fanova', '_'.join(parse_pairwise(p)) + '.png') for p in pairwise_imp.keys()}
        result['Pairwise Marginals'] = {p : {'figure' : path} for p, path in pairwise_plots.items() if os.path.exists(path)}

        return result

    def get_jupyter(self):
        from IPython.core.display import HTML, Image, display
        for b, result in self.result.items():
            error = self.result[b]['else'] if 'else' in self.result[b] else None
            if error:
                display(HTML(error))
            else:
                # Show table
                display(HTML(self.result[b]["Importance"]["table"]))
                # Show plots
                display(*list([Image(filename=d["figure"]) for d in self.result[b]['Marginals'].values()]))
                display(*list([Image(filename=d["figure"]) for d in self.result[b]['Pairwise Marginals'].values()]))
                # While working for a prettier solution, this might be an option:
                #display(HTML(figure_to_html([d["figure"] for d in self.result[b]['Marginals'].values()] +
                #                            [d["figure"] for d in self.result[b]['Pairwise Marginals'].values()],
                #                            max_in_a_row=3, true_break_between_rows=True)))

