Contribute
==========

Development Guide.

A few notes on the general layout of the project (written for version 1.3.4).

Analyzers
---------

CAVE is built in a modular way, so individual analyzer's (an analyzer is a specific analysis with a specific outcome,
such as a plot or a table) can be added or modified easily.
The `analyzer`-module contains all the individual analyzer's, that inherit from the `BaseAnalyzer` (here, the interface
is defined and explained).

Some of the analyzer's are merely wrappers, because the actual method either already existed elsewhere (see the
parameter importance methods, that are imported from automl's `ParameterImportance`) or they are better off in their
own classes (like `ConfiguratorFootprint`) and located in the `plot` module.

Adding new analyzer's consists of a few steps:
  * write the class that inherits from `BaseAnalyzer` (either a wrapper for some other tool or class or an analyzer on it's own)
  * import the class in `CAVE`-class in cavefacade.py and add a `@analyzer_type`-function to call it
  * add it in `CAVE`-classes `analyze`-method
  * add an entry in `utils/options/default_analysis_options.ini`

That's it, now your analyzer is known to CAVE. To access the data, have a look at the function's of the `RunsContainer`,
that will be accessible in your analyzer after you called `super().__init__`.

Check the `BaseAnalyzer` for a general understanding of the functions you should implement. There are several use-cases
of CAVE, but it should generally suffice to implement the `get_html`-method or even just to save your results to the
`self.results`-dictionary.

CAVE will then process it for the html-report and also show it in jupyter-notebooks.

If you built your analyzer on `bokeh`, make sure to separate the actual analysis from the plotting. That way, you can easily regenerate the plots whenever necessary.

New file-formats
----------------

If you want to support a new file-format for the configurator data that is to be analyzed, the easiest way is writing a
converter. Inherit from `reader.conversion.base_converter` and make sure you overwrite all necessary methods. You should
convert all data in `SMAC3`-format.

You will need to edit the format-detection at `utils.helpers.detect_fileformat` to make your fileformat automatically
discoverable.

If there are additional python-objects you want to use in format-specific analysis,
just add it to the dictionary returned by your `convert`-method. The `RunsContainer` will make it available in a
dictionary at `runscontainer.share_information` with the same keyword/value pair.

Alternatively you can also implement a new reader, that you inherit from `reader.base_reader`.
In this case you need to change more code to make your reader available to CAVE - check the SMAC2 example reader for
reference.

Testing
-------

Since most of CAVE's methods include plotting, there is not too many unit-tests.
While surely it would be justified to add more testing, most of CAVE's testing relies on whitebox-testing.
Before any release you should `whitebox_tests/run_examples.sh` to see if there are any obvious errors coming up.
In addition of course `nosetests` will run the testing infrastructure at `tests`.

Keep in mind to also check your modifications in a jupyter-notebook, since there are some caveats when using IPython.
