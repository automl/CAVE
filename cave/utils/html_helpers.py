from pandas import DataFrame

from smac.configspace import Configuration


def compare_configs_to_html(default: Configuration, incumbent: Configuration):
    """Create HTML-table to compare Configurations. Removes unused parameters.

    Parameters
    ----------
    default, incumbent: Configurations
        configurations to be converted

    Returns
    -------
    table: str
        HTML-table comparing default and incumbent
    """
    # Remove unused parameters
    keys = [k for k in default.keys() if default[k] or incumbent[k]]
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
    table = df.to_html()
    return table
