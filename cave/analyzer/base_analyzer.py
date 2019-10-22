import logging
from collections import OrderedDict
from typing import List, Tuple

from bokeh.io import output_notebook
from bokeh.plotting import show

from cave.html.html_builder import HTMLBuilder
from cave.reader.runs_container import RunsContainer


class BaseAnalyzer(object):

    def __init__(self,
                 runscontainer: RunsContainer,
                 *args,
                 **kwargs):
        """
        runscontainer: RunsContainer
        contains all important information about the configurator runs
        """
        self.logger = logging.getLogger(self.__module__ + '.' + self.__class__.__name__)
        self.name = self.__class__.__name__  # Back-up, can be overwritten, will be used as a name for analysis
        self.logger.debug("Initializing %s", self.name)
        self.runscontainer = runscontainer
        self.result = OrderedDict()

    def plot_bokeh(self):
        """ This function needs to be called if bokeh-plots are to be displayed in notebook AND saved to webpage."""
        raise NotImplementedError()

    def get_html(self, d=None, tooltip=None) -> Tuple[str, str]:
        """General reports in html-format, to be easily integrated in html-code. ALSO FOR BOKEH-OUTPUT.

        Parameters
        ----------
        d: Dictionary
            a dictionary that will be later turned into a website

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
        combine =  "\n\n".join([script, div])
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
        result = []  # all bokeh models
        for k, v in d.items():
            if isinstance(v, dict):
                res = cls.check_for_bokeh(v)
                if res:
                    result.extend(res)
            if k == 'bokeh':
                result.append(v)
        return result
