Quickstart
----------
CAVE is invoked via the commandline. After completing the
`Installation <installation.html>`_ , type

.. code-block:: bash

    python scripts/cave --folders examples/smac3/example_output/* --ta_exec_dir examples/smac3/

to start the example (assuming you
cloned the GitHub-repository in which the example is included). By default, CAVE
will execute all parts of the analysis. To disable certain (timeconsuming) parts
of the analysis, please see the section commandline-options.

Supported file-formats for configurator-output are currently the output of
*SMAC2* and *SMAC3*, as well as csv-data formatted as specified below.

*SMAC2*
=======
Relevant files for the analysis of *SMAC2* (relative to the specified
folder with ??? as wildcards for digits) are:
- scenario.txt
- run_and_results-it???.csv
- paramstrings-it???.txt
- ../traj-run-?.txt
plus the files specified in the scenario.txt (pcs_fn.pcs, instance- and
instancefeature-fn.txt, ...)

*SMAC3*
=======
Relevant files for the analysis of *SMAC3* (relative to the specified
folder) are:
- scenario.txt
- runhistory.json
- traj_aclib2.json
- validated_runhistory.json (optional)
plus the files specified in the scenario.txt (pcs_fn.pcs, instance- and
instancefeature-fn.txt, ...)

*CSV*
=====
CSV gives the opportunity to parse all the rundata in a very simple
format. A scenario-file (and therein specified files) is still required.
`runhistory.csv` substitutes the runhistory, each line representing one target
algorithm run. The first line of the file is the header, with the following
entries: `cost`, `time`, `status`, `seed`, `config_id` and `instance_id`.
When specifying `config_id`, the specified folder must also contain a `configurations.csv`.
If you use `instance_id`-columns, the instance_ids must be the same
that are specified in the instance-feature-file.

`instance_features.csv` (path for instance_features is used from provided scenario):

    +-------------+-----------------+-----------------+-----+
    | INSTANCE_ID | inst_feat_name1 | inst_feat_name2 | ... |
    +=============+=================+=================+=====+
    | 0           | value1          | value2          | ... |
    +-------------+-----------------+-----------------+-----+
    | ...         | ...             | ...             | ... |
    +-------------+-----------------+-----------------+-----+

`configurations.csv`:

    +-----------+-----------------+-----------------+-----+
    | CONFIG_ID | parameter_name1 | parameter_name2 | ... |
    +===========+=================+=================+=====+
    | 0         | value1          | value2          | ... |
    +-----------+-----------------+-----------------+-----+
    | ...       | ...             | ...             | ... |
    +-----------+-----------------+-----------------+-----+

`runhistory.csv`:

    +--------------------+--------------------+------+------+------+--------+
    |      config_id     |  instance_id       | cost | time | seed | status |
    +====================+====================+======+======+======+========+
    | name of config 1   | name of instance 1 | ...  |  ... | ...  |  ...   |
    +--------------------+--------------------+------+------+------+--------+
    |         ...        |          ...       | ...  |  ... | ...  |  ...   |
    +--------------------+--------------------+------+------+------+--------+

`trajectory.csv`:

    +----------+------+----------------+-----------+
    | cpu_time | cost | wallclock_time | incumbent |
    +==========+======+================+===========+
    | ...      | ...  | ...            | ...       |
    +----------+------+----------------+-----------+

Alternatively CAVE can also read in one `runhistory.csv`-file containing all the information
about parameters and instances, in this case the file `configurations.csv` is
not needed. See below for example:

`runhistory.csv`:

    +------+------+------+--------+------------+------------+-----+------------+------------+-----+
    | cost | time | seed | status | parameter1 | parameter2 | ... | inst_feat1 | inst_feat2 | ... |
    +======+======+======+========+============+============+=====+============+============+=====+
    | ...  |  ... | ...  |  ...   | ...        | ...        | ... | ...        | ...        | ... |
    +------+------+------+--------+------------+------------+-----+------------+------------+-----+

`trajectory.csv`:

    +----------+------+----------------+------------+------------+-----+
    | cpu_time | cost | wallclock_time | parameter1 | parameter2 | ... |
    +==========+======+================+============+============+=====+
    | ...      | ...  | ...            | ...        | ...        | ... |
    +----------+------+----------------+------------+------------+-----+
