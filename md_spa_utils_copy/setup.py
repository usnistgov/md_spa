#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('CHANGELOG.rst') as history_file:
    history = history_file.read()

requirements = ["numpy", 'sphinx',"sphinx-argparse", "sphinx_rtd_theme", 'sphinx-jsonschema','sphinxcontrib.blockdiag']

setup_requirements = ['pytest-runner', 'wheel', ]

test_requirements = ['pytest>=3',]

setup(
    author="Jennifer A. Clark",
    author_email='jennifer.clark@nist.gov',
    python_requires='>=3.5',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="This package contains general functions used in MD-SPA that might also be applied in other contexts.",
    install_requires=requirements,
    license="BSD license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='md_spa_utils',
    name='md_spa_utils',
    packages=find_packages(include=['md_spa_utils', 'md_spa_utils.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://git@gitlab.nist.gov/jac16/md-spa-utils',
    version='0.0.0',
    zip_safe=False,
)
