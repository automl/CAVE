F.A.Q.
======

.. rubric:: The automatic export of bokeh-plots does not work, why?

You're likely seeing a warning refering to missing installs of selenium and/or
phantomjs-prebuilt. While selenium should be included in the requirements,
phantomjs-prebuilt is not a pip-install. Please refer to the documentation of
phantomjs to see how installation works (usually `npm install -g phantomjs-prebuilt` should do the trick).

.. rubric:: What is the `ta_exec_dir`-option good for?

The `ta_exec_dir` specifies the target algorithm execution directory. With this option you can run *CAVE* from any
folder, just make sure the relative paths in the scenario can be found from the path you specify in `ta_exec_dir`.
