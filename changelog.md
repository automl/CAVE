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
