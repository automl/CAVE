# 1.2.0

## Bug fixes

* Fix tooltips sometimes not showing
* Fix docu-build
* Fix logging (was printing sometimes)
* Fix bug in reading in configuration in CSV-data

## Major changes

* Change internal structure from cave-in-cave to proper data-handling using a RunsContainer with individual ConfiguratorRuns (per budget/parallel-run combination)
* Add options-file in .ini format to increase flexibility on default options (and technically enable using options-file for CAVE, though not supported by cmdline yet, only in python/jupyter)
* Make the `--folders`-cmd line argument positional (though the keyword `--folders` is still supported)
* Enable automatic file-format detection (`--file_format`-cmd line argument still supported)

## Minor changes

* Add whitebox-tests and script to execute all examples
* Add exceptions for Deactivated and NotApplicable errors and load a lot more work onto BaseAnalyzer

# 1.1.8

## Bug fixes

* Fix args-Errors when trying to use with SMAC 0.11.x
* Fix overflow in blcs checkboxes

## Minor changes

* Force pyimp >= 1.0.6

# 1.1.7

## Bug fixes

* Fix error's in some plots related to NaN's in bohb-trajectory
* Fix fanova display error in jupyter notebook
* Fix critical error on rh-reduction in cfp (individual budget's didn't reduce, leading to KeyError)
* Fix too many cfp-plots (for individual budgets) in some cases
* Fix several jupyter-related bugs
* (minor) Fix TypeError when additional-info in bohb-result is a list (not a dict)

## Major changes

* Add option `cot_inc_traj` to choose the way the bohb-trajectory is interpreted (add `racing` method)
* Add ConfigSpace to overview-table
* Budgets with long floats are rounded as much as possible while staying unique
* Add hpbandster to requirements (overdue, this is mainly used for hpbandster)

## Minor changes

* Update bokeh to 1.1.0
* Format issues (dos to unix)

# 1.1.6

## Bug fixes

* Recreate output always, zip old files if present
* Fix jupyter-notebook logging and output-folder recreation

## Minor changes

* Show budgets as integers, if they are integers
* Add single budgets to cost-over-time plot (from bohb-result)
* Add std to bohb-runs in cost-over-time plot
* Remove combined option from contour-plot-selection, fix checkboxes
* Improve budget correlation plot (add dropdown-selection)
* Remove package orange3

# 1.1.5

## Bug fixes

* Parallel Coordinates order now derived from parameter importance
* Show all fANOVA pairwise plots
* Fix bug in cost-over-time if run multiple times with same output-dir
* Fix multiple bugs related to floats in Categoricals
* Fix surface prediction in cfp when forbidden conditions
* Fix path-spec error on loading SMAC-data
* Sub-dict's were not generated in HTML-report sometimes
* Parameters active in either def or inc are shown in comparison (before only active in def)

## Minor changes

* Add number of bohb-runs to meta-table
* Make overflowing HTML-tables scrollable
* Don't plot scatter or cdf if leq than one instance
* Light changes in output-dir-structure
* Evaluate all parameters in importance
* Group cmd-line-args
* 'Cost' instead of 'Quality' label in all plots
* Redesign BOHB-lcs-plot to allow for 100+ iterations

# 1.1.4

## Bug fixes

* Enforcing smac version 0.10.0 to fix pcs-loading issues for booleans
* Catch NaN-values from BOHB-results
* Fix sorting-bug in pimp-table
* Fix notebook-errors when executing cfp

## Major changes

* New version of the overview-table (with run-specific for every run)
* Enable support for multiple BOHB-runs (just aggregating budgets across results)

## Minor changes

* Make nomenclature for BOHB clearer
* Add BOHB to extensive-testing-suite

# 1.1.1

## Major changes

* Change support for BOHB-reports, creating one html-file with all budgets as sub-reports now.
* There are now two ways to generate a report: one uses the given runs as equal, aggregates them and trains an EPM on
  all of them, the other is the use-budgets option that tries to treat them as separate runs, that can only be compared
  in certain fashion
* Fix critical errors in csv- and smac2- reader
* Catch failed bokeh-exports
* Add algorithm footprints mouseover
* Add fanova std's when using bleeding edge of pyimp and fanova

## Interface changes

* passing most of the relevant arguments directly from facade to analyzer now (not during analyzer's __init__)

## Minor changes

* improving documentation and comments, add custom logos
