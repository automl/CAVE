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
