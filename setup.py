"""
This setup file was copied and adapted from the SMAC-project
(https://github.com/automl/SMAC3)
"""

import setuptools

import spysmac


with open('requirements.txt') as fh:
    requirements = fh.read()
requirements = requirements.split('\n')
requirements = [requirement.strip() for requirement in requirements]

with open("spysmac/__version__.py") as fh:
    version = fh.readlines()[-1].split()[-1].strip("\"'")


setuptools.setup(
    name="spysmac",
    version=version,
    author=spysmac.AUTHORS,
    author_email="fh@cs.uni-freiburg.de",
    description=("SpySMAC builds upon SMAC to provide an easy-to-use analysis tool "
                 "for the output of SMAC optimization."),
    license="3-clause BSD",
    keywords="machine learning algorithm configuration hyperparameter "
             "optimization tuning",
    url="",
    packages=setuptools.find_packages(exclude=['test', 'source']),
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
