Input Format
============

Supported file-formats for configurator-output are currently the output of
`SMAC2 <http://www.cs.ubc.ca/labs/beta/Projects/SMAC/>`_, `SMAC3 <https://github.com/automl/SMAC3>`_ and `BOHB <https://github.com/automl/HpBandSter>`_, as well as csv-data formatted as specified below.
The file-format should be detected automatically. If that fails, please report this as an issue with a minimum working example.

*SMAC3*
~~~~~~~
Relevant files for the analysis of *SMAC3* (relative to the specified
folder) are:

- scenario.txt
- runhistory.json
- traj_aclib2.json
- validated_runhistory.json (optional)

plus the files specified in the scenario.txt (pcs_fn.pcs, instance- and
instancefeature-fn.txt, ...)

*BOHB*
~~~~~~
To analyzer BOHB-results, you need to run BOHB with a result-logger (`check
here <https://automl.github.io/HpBandSter/build/html/auto_examples/example_5_mnist.html>`_ for an example on how to do it).
The folder you specify must contain:

- configs.json
- results.json
- configspace.pcs

*configs.json* and *results.json* will be automatically logged by the result-logger. Save a ConfigSpace `config_space`
in a directory `output_dir` using

.. code-block:: python

    from ConfigSpace.read_and_write import json
    with open(output_dir, 'w') as fh:
        fh.write(json(config_space))

*CSV*
~~~~~
Comma-separated values give the opportunity to parse all the rundata in a very simple
format. A scenario-file (and therein specified files) is still required.
`runhistory.csv` substitutes the runhistory, each line representing one target
algorithm run. The first line of the file is the header, with the following
entries: `cost`, `time`, `status`, `seed`, `config_id` and `instance_id`.
Also `budget` is supported, but optional.
When specifying `config_id`, the specified folder must also contain a `configurations.csv`.
If you use `instance_id`-columns, the instance_ids must be the same that are specified in the instance-feature-file.

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

    +--------------------+--------------------+------+------+------+--------+----------+
    |      config_id     |  instance_id       | cost | time | seed | status | (budget) |
    +====================+====================+======+======+======+========+==========+
    | name of config 1   | name of instance 1 | ...  |  ... | ...  |  ...   |  ...     |
    +--------------------+--------------------+------+------+------+--------+----------+
    |         ...        |          ...       | ...  |  ... | ...  |  ...   |  ...     |
    +--------------------+--------------------+------+------+------+--------+----------+

`trajectory.csv`:

    +----------+------+----------------+-----------+----------+
    | cpu_time | cost | wallclock_time | incumbent | (budget) |
    +==========+======+================+===========+==========+
    | ...      | ...  | ...            | ...       | ...      |
    +----------+------+----------------+-----------+----------+

Alternatively CAVE can also read in one `runhistory.csv`-file containing all the information
about parameters and instances, in this case the file `configurations.csv` is
not needed. See below for example:

`runhistory.csv`:

    +------+------+------+--------+----------+------------+------------+-----+------------+------------+-----+
    | cost | time | seed | status | (budget) | parameter1 | parameter2 | ... | inst_feat1 | inst_feat2 | ... |
    +======+======+======+========+==========+============+============+=====+============+============+=====+
    | ...  |  ... | ...  |  ...   | ...      | ...        | ...        | ... | ...        | ...        | ... |
    +------+------+------+--------+----------+------------+------------+-----+------------+------------+-----+

`trajectory.csv`:

    +----------+------+----------------+----------+------------+------------+-----+
    | cpu_time | cost | wallclock_time | (budget) | parameter1 | parameter2 | ... |
    +==========+======+================+==========+============+============+=====+
    | ...      | ...  | ...            | ...      | ...        | ...        | ... |
    +----------+------+----------------+----------+------------+------------+-----+

*APT*
~~~~~

**NOTE** *Auto-PyTorch is still in alpha at the time of writing.
Therefore, this describes more of a sketch and will likely not reflect the current status of development.
The last known working version of Auto-PyTorch is
https://github.com/automl/Auto-PyTorch/commit/a39012ff464a02eead9315a00179812206235f25*

Relevant files for the analysis of *APT* are:

- configs.json, results.json (same format as in *BOHB*)
- configspace.json
- autonet_config.json
- results_fit.json

For an example on how to use it, check
`the notebook <https://github.com/automl/CAVE/blob/master/examples/autopytorch/apt_notebook.ipynb>`_.

*SMAC2*
~~~~~~~

**NOTE** *The SMAC2 format is not extensively tested, but expected to work. If you experience any problems, please report them to CAVE's issue tracker.*

Relevant files for the analysis of *SMAC2* (relative to the specified
folder with ??? as wildcards for digits) are:

- scenario.txt
- run_and_results-it???.csv
- paramstrings-it???.txt
- ../traj-run-?.txt

plus the files specified in the scenario.txt (pcs_fn.pcs, instance- and
instancefeature-fn.txt, ...)
