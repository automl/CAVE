Quickstart
----------
CAVE is invoked via the commandline. After completing the
`Installation <installation.html>`_ , type
.. code-block:: bash

    python scripts/cave --folders examples/spear_qcp_small/example_output/* --ta_exec_dir examples/spear_qcp_small/

to start the example (assuming you
cloned the GitHub-repository in which the example is included). By default, CAVE
will execute all parts of the analysis. To disable certain (timeconsuming) parts
of the analysis, please see the section commandline-options.

Supported file-formats for configurator-output are currently the output of
*SMAC2* and *SMAC3*, as well as csv-data formatted as specified below.

*SMAC2*: Relevant files for the analysis of *SMAC2* (relative to the specified
folder with ??? as wildcards for digits) are:
- scenario.txt
- run_and_results-it???.csv
- paramstrings-it???.txt
- ../traj-run-?.txt
plus the files specified in the scenario.txt (pcs_fn.pcs, instance- and
instancefeature-fn.txt, ...)

*SMAC3*: Relevant files for the analysis of *SMAC3* (relative to the specified
folder) are:
- scenario.txt
- runhistory.json
- traj_aclib2.json
- validated_runhistory.json (optional)
plus the files specified in the scenario.txt (pcs_fn.pcs, instance- and
instancefeature-fn.txt, ...)

*CSV*: CSV gives the opportunity to parse all the rundata in a very simple
format. A scenario-file (and therein specified files) is still required.
`runhistory.csv` substitutes the runhistory, each line representing one target
algorithm run. The first line of the file is the header, with the following
entries: `cost`, `time`, `status`, `seed`. Everything else in that row will be
interpreted as parameter-values or instance-feature-values. The
interpretation-topology is as follows:
If there is a PCS given, the names of the parameters will be used to identify
parameter-columns and every unidentified column will be an
instance-feature-column.
If there is no PCS given, but an instance-feature-file, the names of the
instance-features will be used to identify instance-feature-columns and all the
others are parameters.
If neither PCS nor instance-features are specified, everything will be
interpreted as parameters.

     +--------------------+--------------------+------------------------+------+
     |      config_id     |  instance_id       | cost                   | seed |
     +====================+====================+========================+======+
     | name of config 1   | name of instance 1 | cost of this row's run | ...  |
     +--------------------+--------------------+------------------------+------+
     |         ...        |          ...       |          ...           | ...  |
     +--------------------+--------------------+------------------------+------+

As an alternative to specifiying the configurations and instance-features
directly in the `runhistory.csv`, it is possible to use the columns `config_id`
and/or `instance_id`. When specifying `config_id`, the specified folder must
also contain a `configurations.csv` with the header `CONFIG_ID, parameter1,
parameter2, ...`. If you use `instance_id`-columns, the instance_ids must be the same
that are specified in the instance-feature-file.
