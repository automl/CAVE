import setuptools

import cave


with open('requirements.txt') as fh:
    requirements = fh.read()
requirements = requirements.split('\n')
requirements = [requirement.strip() for requirement in requirements
                  if not "http" in requirement]

with open("cave/__version__.py") as fh:
    version = fh.readlines()[-1].split()[-1].strip("\"'")


setuptools.setup(
    name="cave",
    version=version,
    author=cave.AUTHORS,
    # TODO author email
    author_email="",
    description=("CAVE, an analyzing tool for configuration optimizers"),
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
    tests_require=['mock',
                   'nose'],
    test_suite='nose.collector'
)
