#!/usr/bin/env python

"""The setup script."""

import os
import glob
from distutils.core import setup, Extension
from setuptools import setup, find_packages

# Find and prepare cython modules
try:
    import numpy as np
    from Cython.Build import cythonize
    flag_cython = True
except Exception:
    print(
        'Cython not available on your system. Proceeding without C-extensions.'
    )
    flag_cython = False
if flag_cython:
    fpath = os.path.join("md_spa", "cython_modules")
    cython_list = glob.glob(os.path.join(fpath, "*.pyx"))
    extensions = []
    for cyext in cython_list:
        cypath = list(os.path.split(cyext))
        cypath[-1] = cypath[-1].split(".")[-2]
        cy_ext_1 = Extension(name=".".join(cypath), 
                             sources=[cyext], 
                             include_dirs=[fpath, np.get_include()],
                             extra_compile_args=['-Wno-deprecated-declarations'],
                            )
        extensions.extend(
            cythonize([cy_ext_1],
                      compiler_directives={
                          'language_level': 3,
                          'cdivision': False,
                          "boundscheck": True
                      }))

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('CHANGELOG.rst') as history_file:
    history = history_file.read()

requirements = [
                'sphinx',
                "sphinx-argparse",
                "sphinx_rtd_theme",
                'sphinx-jsonschema',
                'sphinxcontrib.blockdiag',
                'numpy',
                'scipy',
                'lmfit',
                'matplotlib',
                'md_spa_utils',
                ]

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
    description="Molecular Dynamics Simulation Properties Analysis (MD-SPA) will ease the extraction of relevant property information.",
    install_requires=requirements,
    license="BSD license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='md_spa',
    name='md_spa',
    packages=find_packages(include=['md_spa', 'md_spa.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://git@gitlab.nist.gov/jac16/md-spa',
    version='0.0.0',
    ext_modules=extensions,
    extras_require={
        "tests": ["pytest"],
    },
    zip_safe=False,
)
