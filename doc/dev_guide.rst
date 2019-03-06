Developer's Guide
-----------------
CAVE aims to be modular and easily extendable. This section summarizes the most important concepts behind the
architecture.

Custom Analyzers
~~~~~~~~~~~~~~~~
To write a custom analyzer, you need to inherit from `the BaseAnalyzer <apidoc/cave.analyzer.base_analyzer.html>`_ 


While it's possible to generate a HTML-report via the commandline on a
given result-folder, CAVE may also run in interactive mode, running `individual analysis-methods <apidoc/cave.cavefacade.html>`_ on demand. We provide a
few examples to demonstrate this.
Make sure you followed the `installation details <installation.html>`_ before starting.

Analyse existing results via the commandline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are example toy results of all supported file formats in the folder `examples
<https://github.com/automl/CAVE/tree/master/examples>`_ on the github-repo.
Run

.. code-block:: bash

    cave --folders examples/smac3/example_output/* --ta_exec_dir examples/smac3/ --output CAVE_OUTPUT

to start the example (assuming you cloned the GitHub-repository in which the example is included).
By default, CAVE will execute all parts of the analysis. To disable certain (timeconsuming) parts
