import argparse
import logging
import sys
import os
import shutil
import inspect
import re
from traceback import print_exc
from collections import namedtuple

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from cave.utils.tooltips import get_tooltip

__author__ = "Marius Lindauer"
__copyright__ = "Copyright 2016, ML4AAD"
__license__ = "MIT"
__email__ = "lindauer@cs.uni-freiburg.de"

class HTMLBuilder(object):

    def __init__(self,
                 output_dn:str,
                 scenario_name:str):
        '''
        Constructor

        Arguments
        ---------
        output_dn:str
            output directory name
        scenario_name:str
            name of scenario
        '''
        self.logger = logging.getLogger("HTMLBuilder")


        self.own_folder = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ))[0]))

        self.output_dn = output_dn
        self.relative_content_js = os.path.join('content', 'js')
        self.relative_content_images = os.path.join('content', 'images')
        os.makedirs(os.path.join(self.output_dn, self.relative_content_js), exist_ok=True)
        os.makedirs(os.path.join(self.output_dn, self.relative_content_images), exist_ok=True)

        self.header_part_1 = '''
<!DOCTYPE html>
<html>
<head>
<title>CAVE</title>
<link href="html/css/accordion.css" rel="stylesheet" />
<link href="html/css/table.css" rel="stylesheet" />
<link href="html/css/lightbox.min.css" rel="stylesheet" />
<link href="html/css/help-tip.css" rel="stylesheet" />
<link href="html/css/global.css" rel="stylesheet" />
<link href="html/css/back-to-top.css" rel="stylesheet" />

<link href="html/css/bokeh-0.12.14.min.css" rel="stylesheet" type="text/css">
<link href="html/css/bokeh-widgets-0.12.14.min.css" rel="stylesheet" type="text/css">
<link href="html/css/bokeh-tables-0.12.14.min.css" rel="stylesheet" type="text/css">

<script src="html/js/bokeh-0.12.14.min.js"></script>
<script src="html/js/bokeh-widgets-0.12.14.min.js"></script>
<script src="html/js/bokeh-tables-0.12.14.min.js"></script>

<!--Below here are the includes of scripts for the report (e.g. bokeh)-->
'''

        self.header_part_2 = '''
<!--Above here are the includes of scripts for the report (e.g. bokeh)-->

</head>
<body>
<script src="http://www.w3schools.com/lib/w3data.js"></script>
<script src="html/js/lightbox-plus-jquery.min.js"></script>
<header>
    <div class='l-wrapper'>
        <img class='logo logo--smac3' src="html/images/SMAC3.png" />
        <img class='logo logo--ml' src="html/images/ml4aad.png" />
    </div>
</header>
<div class='l-wrapper'>
<h1>CAVE</h1>
'''

        self.footer = '''
</div>
<footer>
    <div class='l-wrapper'>
        Generated by <a href="https://github.com/automl/CAVE">CAVE</a> and developed by <a href="http://www.ml4aad.org">ML4AAD</a> | Optimized for Chrome and Firefox
    </div>
</footer>
<script>
var acc = document.getElementsByClassName("accordion");
var i;
for (i = 0; i < acc.length; i++) {
    acc[i].onclick = function(){
        this.classList.toggle("active");
        this.nextElementSibling.classList.toggle("show");
  }
}
</script>
<script src="html/js/back-to-top.js"></script>
</body>
</html>
'''

    def generate_html(self, data_dict:dict):
        '''
        Arguments
        ---------
        data_dict : OrderedDict
            {"top1" : {
                        "tooltip": str|None,
                        "subtop1: {  # generates a further bottom if it is dictionary
                                "tooltip": str|None,
                                ...
                                }
                        "table": str|None (html table)
                        "figure" : str|None (file name)
                        "bokeh" : ( str,str)|None  # (script, div)
                        }
            "top2: { ... }
        '''
        html_head, html_body = "", ""
        html_head += self.header_part_1
        html_dict = {}
        # Get components (script, div) for each entry in report
        for k, v in data_dict.items():
            script, div = self.add_layer(layer_name=k, data_dict=v)
            html_dict[k] = {'script' : script, 'div' : div}
        # Scripts go into header, divs go into body
        for k, v in html_dict.items():
            if v['script']:
                html_head += v['script']  # e.g. bokeh-scripts used for hover
            html_body += v['div']
        html_head += self.header_part_2  # Close header after adding all scripts
        html = html_head + html_body + self.footer

        with open(os.path.join(self.output_dn, "report.html"), "w") as fp:
            fp.write(html)

        subfolders = ["css", "images", "js", "font"]
        for sf in subfolders:
            try:
                if not os.path.isdir(os.path.join(self.output_dn, "html", sf)):
                    shutil.copytree(os.path.join(self.own_folder, "web_files", sf),
                                    os.path.join(self.output_dn, "html", sf))
            except OSError:
                print_exc()


    def add_layer(self, layer_name, data_dict:dict):
        '''
        add a further layer of top data_dict keys

        Parameters
        ----------
        layer_name: str
            name of the layer
        data_dict : OrderedDict
            {"top1" : {
                        "tooltip": str|None,
                        "subtop1: {  # generates a further bottom if it is dictionary
                                "tooltip": str|None,
                                ...
                                }
                        "table": str|None (html table)
                        "figure" : str|None (file name)
                        "bokeh" : ( str,str)|None  # (script, div)
                        }
            "top2: { ... }

        Returns
        -------
        (script, div): (str, str)
            script goes into header, div goes into body
        '''
        script, div = "", ""
        if data_dict.get("tooltip"):
            tooltip = "<div class=\"help-tip\"><p>{}</p></div>".format(data_dict.get("tooltip"))
        elif get_tooltip(layer_name):  # if no tooltip is parsed, try to look it up
            tooltip = "<div class=\"help-tip\"><p>{}</p></div>".format(get_tooltip(layer_name))
        else:
            tooltip = ""
        div += "<div class=\"accordion\">{0} {1}</div>\n".format(layer_name, tooltip)
        div += "<div class=\"panel\">\n"

        for k, v in data_dict.items():
            if isinstance(v, dict):
                add_script, add_div = self.add_layer(k, v)
                script += add_script
                div += add_div
            elif k == "figure":
                div += "<div align=\"center\">\n"
                if isinstance(v, str):
                    div += ("<a href=\"{0}\" data-lightbox=\"{0}\" "
                            "data-title=\"{0}\"><img src=\"{0}\" alt=\"Plot\" "
                            "width=\"600px\"></a>\n".format(v[len(self.output_dn):].lstrip("/")))
                else:
                    # List with multiple figures size relative, put next to each other
                    width = (100 - len(v)) / len(v)
                    for fig in v:
                        div += "<a href=\"{0}\" data-lightbox=\"{1}\" data-title=\"{0}\"><img src=\"{0}\" alt=\"Plot\" style=\"float: left; width: {2}%; margin-right: 1%; margin-bottom: 0.5em;\"></a>\n".format(
                                fig[len(self.output_dn):].lstrip("/"),
                                str(v),
                                int(width))

                    div += "<p style=\"clear: both;\">"
                div += "</div>\n"
            elif k == "figure_x2":
                # four figures in a grid
                div += "<div align=\"center\">\n"
                for fig in v:
                    path = fig[len(self.output_dn):].lstrip("/")
                    div += "<a href=\"{}\" ".format(path)
                    div += "data-lightbox=\"{}\" ".format(str(v))
                    div += "data-title=\"{0}\"><img src=\"{0}\" alt=\"Plot\" ".format(path)
                    div += "style=\"float: left; width: 49%; margin-right: 1%; margin-bottom: 0.5em;\"></a>\n"
                    if v.index(fig) % 2 == 1:
                        div += " <br> "

                div += "<p style=\"clear: both;\">"
                div += "</div>\n"
            elif k == "table":
                div += "<div align=\"center\">\n{}\n</div>\n".format(v)
            elif k == "html":
                div += ("<div align=\"center\">\n<a href='{}'>Interactive "
                        "Plot</a>\n</div>\n".format(v[len(self.output_dn):].lstrip("/")))
            elif k == "bokeh":
                # Escape path for URL (replace   and ' with   . ;)
                path_script = os.path.join(self.relative_content_js, layer_name + '_script.js')
                path_script = path_script.translate({ord(c): None for c in ' \''})
                # Write script to file
                with open(os.path.join(self.output_dn, path_script), 'w') as fn:
                    js_code = re.sub('<.*?>','', v[0].strip())  # Remove script-tags
                    fn.write(js_code)
                script += "<script src=\"" + path_script + "\"></script>\n"
                div += "<div align=\"center\">\n{}\n</div>\n".format(v[1])

        div += "</div>"
        return script, div
