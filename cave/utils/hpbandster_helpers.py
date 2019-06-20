"""
Here are helper functions needed to provide a certain behaviour of HpBandSter, such as special trajectories.
"""

import numpy as np
from collections import OrderedDict

def format_budgets(budgets):
    """
    Format budget-strings so that they are as short as possible while still distinguishable

    Parameters
    ----------
    budgets: List[str]
        list with budgets

    Returns
    -------
    formatted_budgets: List[str]
        list with formatted budgets
    """
    def format_budget(b, round_to):
        return 'budget_{}'.format(int(b)) if float(b).is_integer() else 'budget_{:.{}f}'.format(b, round_to)

    round_to = 1
    formatted_budgets = {b : format_budget(b, round_to) for b in budgets}
    while len(set(formatted_budgets.values())) != len(formatted_budgets.values()):
        round_to += 1
        formatted_budgets = {b : format_budget(b, round_to) for b in budgets}

    return formatted_budgets

def get_incumbent_trajectory(result, budgets, mode='racing'):
    """
    Parameters
    ----------
    result: hpbandster.core.result.Result
        result object
    budgets: List[str|int|float] or 'all'
        if a list of budgets, only consider those budgets (to enable trajectories for only a single budget)
    mode: str
        from ['racing', 'minimum', 'prefer_higher_budget']

    Returns
    -------
    trajectory: dict
        dictionary with all the config IDs, the times the runs finished, their respective budgets, and corresponding
        losses
    """
    if budgets == 'all':
        budgets = list(result.HB_config['budgets'])
    if not isinstance(budgets, list):
        raise ValueError("%s not a valid argument for 'budgets'" % str(budgets))

    if mode == 'racing':
        # Philipp's method
        all_runs = result.get_all_runs(only_largest_budget=False)
        all_runs = list(filter(lambda r: r.budget in budgets, all_runs))
        all_runs.sort(key=lambda run: run.time_stamps['finished'])  # ensure that all_runs and budgets is sorted in ascending order
        budgets.sort()

        return_dict = {'config_ids': [],
                       'times_finished': [],
                       'budgets': [],
                       'losses': []}
        last_run = None
        for r in _compute_trajectory_racing(all_runs, budgets):
            if last_run is None or last_run is not r:
                return_dict['config_ids'].append(r.config_id)
                return_dict['times_finished'].append(r.time_stamps['finished'])
                return_dict['budgets'].append(r.budget)
                return_dict['losses'].append(r.loss)
            last_run = r
        return return_dict
    else:
        # HpBandSter's method (adapted for single budgets)
        if mode == 'minimum':
            return _get_incumbent_trajectory_hpbandster(result, budgets, bigger_is_better=False,
                                                        non_decreasing_budget=False)
        elif mode == 'prefer_higher_budget':
            return _get_incumbent_trajectory_hpbandster(result, budgets, bigger_is_better=True,
                                                        non_decreasing_budget=True)
        else:
            raise ValueError("'%s' not a supported method for get_incumbent_trajectory" % mode)



def _compute_trajectory_racing(all_runs, budgets):
    """
    Computes the trajectory in racing mode.
    If there is at time T a current incumbent I on budget B, a configuration C
    can become a incumbent on a higher budget only if I is also evaluated on C.
    This means a configuration C can only be incumbent if the current incumbent
    is evaluated on the same budget.

    If the current incumbent is not available for a higher budget, the incumbent
    candidate for this higher budget is chosen as incumbent.

    If a configuration 'waits' for the incumbent to be evaluated on the same
    budget, its finishing time is set to the time the old incumbent is evaluated
    on the higher budget.

    Args:
        all_runs : list(hpbandster.core.Run)
            list of all hpbandster runs sorted ascending by its finishing time
        budgets : list(float)
            list of all budgets in ascending order

    Returns : Generator
        Generator representing the incumbents in order of appearance
    """
    if len(all_runs) == 0:
        return []
    # The current incumbents for all budgets
    # - key is the budget and value is the incumbent
    incumbents = OrderedDict()

    # The configurations seen for each budget
    seen = OrderedDict([(x, set()) for x in budgets])
    # The configurations which are available for each budget.
    # It is used to perform a lookup if a configuration is actually evaluated
    # on a budget.
    upcoming = OrderedDict([(x, set()) for x in budgets])
    for a in all_runs:
        upcoming[a.budget].add(a.config_id)

    # The current incumbent to observe
    current_incumbent_budget = budgets[0]

    # Iterate over all timestamps (ordered)
    for run in all_runs:
        if run.loss is None or not np.isfinite(run.loss):
            continue

        # Add config id to seen list for this budget
        seen[run.budget].add(run.config_id)

        # Update incumbent for budget if for a budget no configuration has been
        # seen or the loss is smaller than the current incumbent on this budget
        if run.budget not in incumbents \
                or run.loss < incumbents[run.budget].loss:
            incumbents[run.budget] = run

        # Make a forward check if the current incumbent is actually available
        # on the higher budget. Set the incumbent candidate of the higher budget
        # to the current incumbent if the incumbent on the current budget is not
        # available on the next budget
        if budgets.index(run.budget) == budgets.index(current_incumbent_budget) + 1 \
                 and incumbents[current_incumbent_budget].config_id not in \
                 upcoming[run.budget]:
            current_incumbent_budget = run.budget

        # If there are potential candidates for the higher budgets, a change of
        # the incumbent on the current budget could lead to a series of new
        # incumbents on the higher budgets. Therefore, iterate over the higher
        # budgets and check if there are new incumbents
        if run.config_id == incumbents[current_incumbent_budget].config_id:
            # Go through all higher budgets in order
            for next_budget in [x for x in budgets if x > current_incumbent_budget]:
                # Check if changing the last incumbent is also tested on a
                # higher budget. Break if it is not
                if next_budget not in incumbents or \
                        (incumbents[current_incumbent_budget].config_id not in seen[next_budget]
                         and incumbents[current_incumbent_budget].config_id in upcoming[next_budget]):
                        break
                # Increase the budget to be shown
                current_incumbent_budget = next_budget

        # Update the finished time to the time of setting as incumbent
        incumbents[current_incumbent_budget].time_stamps['finished'] = run.time_stamps['finished']
        yield incumbents[current_incumbent_budget]


def _get_incumbent_trajectory_hpbandster(result, budgets, bigger_is_better=True, non_decreasing_budget=True):
    """
    Returns the best configurations over time

    !! Copied from hpbandster and modified to enable getting trajectories for individual budgets !!

    Parameters
    ----------
        result:
                result
        budgets: List[budgets] or 'all' or 'only_largest'
                budgets to be considered
        bigger_is_better:bool
                flag whether an evaluation on a larger budget is always considered better.
                If True, the incumbent might increase for the first evaluations on a bigger budget
        non_decreasing_budget: bool
                flag whether the budget of a new incumbent should be at least as big as the one for
                the current incumbent.
    Returns
    -------
        dict:
                dictionary with all the config IDs, the times the runs
                finished, their respective budgets, and corresponding losses
    """
    all_runs = result.get_all_runs(only_largest_budget=False)

    if isinstance(budgets, list):
        all_runs = list(filter(lambda r: r.budget in budgets, all_runs))
    elif budgets == 'all':
        pass
    elif budgets == 'only_largest':
        all_runs = result.get_all_runs(only_largest_budget=True)

    all_runs.sort(key=lambda r: r.time_stamps['finished'])

    return_dict = { 'config_ids' : [],
                    'times_finished': [],
                    'budgets'    : [],
                    'losses'     : [],
    }

    current_incumbent = float('inf')
    incumbent_budget = result.HB_config['min_budget']

    for r in all_runs:
        if r.loss is None or not np.isfinite(r.loss):
            continue

        new_incumbent = False

        if bigger_is_better and r.budget > incumbent_budget:
            new_incumbent = True

        if r.loss < current_incumbent:
            new_incumbent = True

        if non_decreasing_budget and r.budget < incumbent_budget:
            new_incumbent = False

        if new_incumbent:
            current_incumbent = r.loss
            incumbent_budget  = r.budget

            return_dict['config_ids'].append(r.config_id)
            return_dict['times_finished'].append(r.time_stamps['finished'])
            return_dict['budgets'].append(r.budget)
            return_dict['losses'].append(r.loss)

    if current_incumbent != r.loss:
        r = all_runs[-1]

        return_dict['config_ids'].append(return_dict['config_ids'][-1])
        return_dict['times_finished'].append(r.time_stamps['finished'])
        return_dict['budgets'].append(return_dict['budgets'][-1])
        return_dict['losses'].append(return_dict['losses'][-1])

    return (return_dict)
