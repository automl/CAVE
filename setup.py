import setuptools

import spysmac


with open('requirements.txt') as fh:
    requirements = fh.read()
requirements = requirements.split('\n')
requirements = [requirement.strip() for requirement in requirements
                  if not "http" in requirement]
#requirements.extend(["smac", "pimp", "fanova"])

with open("spysmac/__version__.py") as fh:
    version = fh.readlines()[-1].split()[-1].strip("\"'")


setuptools.setup(
    name="smac",
    version=version,
    author=spysmac.AUTHORS,
    # TODO author email
    author_email="",
    description=("SpySMAC, an analyzing tool for SMAC3"),
    license="3-clause BSD",
    keywords="machine learning algorithm configuration hyperparameter "
             "optimization tuning analyzing",
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
#    dependency_links=[
#        'https://github.com/automl/fanova@b5d69a7db458f61b3a31e8fbb66edef6d8fce35f#fanova',
#        'https://github.com/automl/SMAC3.git@e641576403e9de1a1188856b5f48e7232ac0d517#egg=smac',
#        'https://github.com/automl/ParameterImportance.git@development#egg=pimp'],
    tests_require=['mock',
                   'nose'],
    test_suite='nose.collector'
)
