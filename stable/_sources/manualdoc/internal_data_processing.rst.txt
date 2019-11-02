Internal Data Processing
~~~~~~~~~~~~~~~~~~~~~~~~

*CAVE* processes multiple results in one analysis in two dimensions:
  - *parallel runs* denotes multiple runs with the same preliminaries, scenarios and only different results
  - *budgets* (aka fidelities) denotes runs with different approximations of the target problem.

Both of these dimensions can be explored at the same time. How *CAVE* interprets the data depends on the
*file_format* of the optimization results. For *BOHB* it will automatically read in budgets, for *SMAC* there
is (at the time of writing) only parallel runs.

Internally, *CAVE* stores a `ConfiguratorRun <apidoc/cave.reader.configurator_run>`_ for each budget-level for each parallel run.
If there are n budgets and m parallel runs, we'll end up with n*m *ConfiguratorRuns* which are stored in a `RunsContainer <apidoc/cave.reader.runs_container>`_.
