import os
import shutil
import subprocess
import sys
import traceback
import setuptools
from setuptools.command.install import install

import cave

with open('requirements.txt') as fh:
    requirements = fh.read()
requirements = requirements.split('\n')
requirements = [requirement.strip() for requirement in requirements
                if not "http" in requirement]

with open("cave/__version__.py") as fh:
    version = fh.readlines()[-1].split()[-1].strip("\"'")

WEB_FILES_LOCATION = os.path.join(os.path.dirname(__file__), 'cave/html/web_files')

setuptools.setup(
    name="cave",
    version=version,
    packages=['cave', 'cave.analyzer', 'cave.feature_analysis', 'cave.reader', 'cave.reader.conversion', 'cave.html',
              'cave.plot', 'cave.plot.parallel_plot', 'cave.utils'],
    # For info about package_data, please check MANIFEST.in (https://stackoverflow.com/questions/7522250/how-to-include-package-data-with-setuptools-distribute)
    package_data={
        'cave/analyzer': [
            'cave/analyzer/mpl_style'
        ],
        'cave/plot': [
            'cave/plot/mpl_style'
        ],
        'cave/plot/parallel_plot': [
            'cave/plot/parallel_plot/parallel_selection_tool.ts',
            'cave/plot/parallel_plot/parallel_reset.ts',
        ],
        'cave/utils' : [
            'cave/utils/options/default_analysis_options.ini',
            'cave/utils/options/default_bohb_analysis_options.ini',
        ],
        'cave/html': [
            'cave/html/web_files/css/accordion.css',
            'cave/html/web_files/css/back-to-top.css',
            'cave/html/web_files/css/bokeh-1.1.0.min.css',
            'cave/html/web_files/css/bokeh-tables-1.1.0.min.css',
            'cave/html/web_files/css/bokeh-widgets-1.1.0.min.css',
            'cave/html/web_files/css/global.css',
            'cave/html/web_files/css/help-tip.css',
            'cave/html/web_files/css/lightbox.min.css',
            'cave/html/web_files/css/table.css',

            'cave/html/web_files/font/fontello/config.json',
            'cave/html/web_files/font/fontello/fontello.eot',
            'cave/html/web_files/font/fontello/fontello.svg',
            'cave/html/web_files/font/fontello/fontello.ttf',
            'cave/html/web_files/font/fontello/fontello.woff',
            'cave/html/web_files/font/fontello/fontello.woff2',

            'cave/html/web_files/images/close.png',
            'cave/html/web_files/images/COSEAL_small.png',
            'cave/html/web_files/images/loading.png',
            'cave/html/web_files/images/ml4aad.png',
            'cave/html/web_files/images/next.png',
            'cave/html/web_files/images/prev.png',
            'cave/html/web_files/images/CAVE_logo.png',
            'cave/html/web_files/images/SMAC_logo.png',
            'cave/html/web_files/images/BOHB_logo.png',
            'cave/html/web_files/images/automl-logo.png',
            'cave/html/web_files/images/close.png',

            'cave/html/web_files/js/back-to-top.js',
            'cave/html/web_files/css/bokeh-1.1.0.min.js',
            'cave/html/web_files/css/bokeh-tables-1.1.0.min.js',
            'cave/html/web_files/css/bokeh-widgets-1.1.0.min.js',
            'cave/html/web_files/js/lightbox-plus-jquery.min.js',
        ]},
    include_package_data=True,
    author=cave.AUTHORS,
    author_email="biedenka@cs.uni-freiburg.de",
    description=("CAVE, an analyzing tool for configuration optimizers"),
    license="3-clause BSD",
    keywords="machine learning algorithm configuration hyperparameter "
             "optimization tuning analyzing analysis visualization",
    url="",
    entry_points={'console_scripts': ['explore-cave=cave.cave_cli:entry_point',
                                      'cave=cave.cave_cli:entry_point']},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: BSD License",
    ],
    platforms=['Linux'],
    install_requires=requirements,
    tests_require=['mock',
                   'nose'],
    test_suite='nose.collector'
)
