import logging
from collections import OrderedDict
from typing import Tuple

from bokeh.io import output_notebook
from bokeh.plotting import show

from cave.html.html_builder import HTMLBuilder
from cave.reader.runs_container import RunsContainer
from cave.utils.exceptions import Deactivated


class BaseAnalyzer(object):
    """
    The base class for analyzing methods. To create a new analyzer, inherit from this class and extend.
    If you already have an analyzer, but need a wrapper to call it, also inherit it from this class.
    You should overwrite the "get_name"-method.
    Currently the initialization calls the analysis. After the analyzer ran, the results should be saved to the member
    self.result, which is a dictionary with a defined structure.
    The docstrings (this part) will be used to display a tooltip / help for the analyzer, so it should be a descriptive
    and concise small paragraph describing the analyzer and it's results.
    Remember to call super.__init__(runscontainer) in your analyzer's __init__-method. This will initialize the logger,
    name and important attributes.
    All configurator data is available via the self.runscontainer.
    """
    def __init__(self,
                 runscontainer: RunsContainer,
                 *args,
                 **kwargs):
        """
        runscontainer: RunsContainer
            contains all important information about the configurator runs
        """
        self.logger = logging.getLogger(self.__module__ + '.' + self.__class__.__name__)
        self.name = self.get_name()
        self.logger.debug("Initializing %s", self.name)
        self.runscontainer = runscontainer
        self.result = OrderedDict()
        self.error = False

        options = self.runscontainer.analyzing_options
        if self.name not in options.sections():
            self.logger.warning("Please state in the analyzing options whether or not to run this Analyzer "
                                "(simply add a line to the .ini file containing [{}])".format(self.name))
        elif not options[self.name].getboolean('run'):
            raise Deactivated("{0} has been deactivated in the options. To enable, just set "
                              "[{0}][run] = True in the .ini file or pass the appropriate flags.".format(self.name))

        self.options = options[self.name]
        for k, v in kwargs.items():
            if v is not None:
                self.options[k] = v
        self.logger.debug("{} initialized with options: {}".format(self.name, str(dict(self.options))))

    def plot_bokeh(self):
        """
        This function should recreate the bokeh-plot from scratch with as little overhead as possible. This is needed to
        show the bokeh plot in jupyter AND save it to the webpage. The bokeh plot needs to be recreated to be displayed
        in different outputs for reasons beyond out control. So save all analysis results in the class and simply redo
        the plotting with this function.
        This function needs to be called if bokeh-plots are to be displayed in notebook AND saved to html-result.
        """
        raise NotImplementedError()

    def get_html(self, d=None, tooltip=None) -> Tuple[str, str]:
        """General reports in html-format, to be easily integrated in html-code. WORKS ALSO FOR BOKEH-OUTPUT.

        Parameters
        ----------
        d: Dictionary
            a dictionary that will be later turned into a website
        tooltip: string
            tooltip to be displayed in report. optional, will overwrite the docstrings that are used by default.

        Returns
        -------
        script, div: str, str
            header and body part of html-code
        """
        if len(self.result) == 1 and None in self.result:
            self.logger.debug("Detected None-key, abstracting away...")
            self.result = self.result[None]
        if d is not None:
            d[self.name] = self.result
            d[self.name]['tooltip'] = tooltip if tooltip is not None else self.__doc__
        script, div = HTMLBuilder("", "", "").add_layer(None, self.result)
        combine = "\n\n".join([script, div])
        return combine

    def get_jupyter(self):
        """Depending on analysis, this creates jupyter-notebook compatible output."""
        bokeh_plots = self.check_for_bokeh(self.result)
        if bokeh_plots:
            self.logger.warning("Bokeh plots cannot be re-used for notebook if they've already been \"components\"'ed. "
                                "To be sure, get_jupyter should be overwritten for bokeh-producing analyzers.")
            output_notebook()
            for bokeh_plot in bokeh_plots:
                show(bokeh_plot)
        else:
            from IPython.core.display import HTML, display
            display(HTML(self.get_html()))

    @classmethod
    def check_for_bokeh(cls, d):
        """
        Check if there is bokeh-plots in the output of this analyzer by checking the result-dictionary for the bokeh
        keyword.
        """
        result = []  # all bokeh models
        for k, v in d.items():
            if isinstance(v, dict):
                res = cls.check_for_bokeh(v)
                if res:
                    result.extend(res)
            if k == 'bokeh':
                result.append(v)
        return result

    def get_name(self):
        return self.__class__.__name__  # Back-up, can be overwritten, will be used as a name for analysis
