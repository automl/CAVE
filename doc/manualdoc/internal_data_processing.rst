Internal Data Processing
~~~~~~~~~~~~~~~~~~~~~~~~

*CAVE* processes multiple results in one analysis in two dimensions:
  - *parallel runs* denotes multiple runs with the same preliminaries, scenarios and only different results
  - *budgets* (aka fidelities) denotes runs with different approximations of the target problem.

Internally, *CAVE* stores a `ConfiguratorRun <apidoc/cave.reader.configurator_run>`_ for each parallel run.
*ConfiguratorRuns* are stored in a `RunsContainer <apidoc/cave.reader.runs_container>`_, which provides easy
methods for aggregation or reduction of *ConfiguratorRuns*.
