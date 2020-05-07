Budgets
~~~~~~~

Some configurators such as `BOHB/hpbandster <https://github.com/automl/HpBandSter>`_ and
`SMAC <https://github.com/automl/SMAC3>`_ > 0.12.0 support optimization using
budgets (aka fidelities). Budgets are limited resources that speed up the evaluation of a configuration, acting as an estimate.
When evaluating neural networks with a reduced number of epochs, the number of epochs is the budget. CAVE
can process and analyze these results. It is possible to access each method for each budget individually.
Also, there are methods to compare across budgets.  CAVE will automatically use budgets, when they are available.
