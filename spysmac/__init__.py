import sys

if sys.version_info < (3,4):
    raise ValueError("SpySMAC requires Python 3.4 or newer.")

from spysmac.__version__ import __version__
AUTHORS = "Marius Lindauer, Joshua Marben" # Matthias Feurer, Katharina Eggensperger, " \
                                           # "Aaron Klein, Stefan Falkner and Frank Hutter"
                                           # TODO how does authorship work, whom
                                           # to include? ...
