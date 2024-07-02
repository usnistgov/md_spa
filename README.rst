======
MD-SPA
======

..
    .. image:: https://git@gitlab.nist.gov/jac16/md-spa/badges/master/pipeline.svg
        :target: https://git@gitlab.nist.gov/jac16/md-spa/pipelines/
        :alt: Build Status
    .. image:: https://git@gitlab.nist.gov/jac16/md-spa/badges/master/coverage.svg
        :target: https://git@gitlab.nist.gov/jac16/md-spa/pipelines/
        :alt: Coverage
    .. image:: https://img.shields.io/badge/License-BSD license-blue.svg
        :target: https://git@gitlab.nist.gov/jac16/md-spa/-/blob/master/LICENSE

Molecular Dynamics Simulation Properties Analysis (MD-SPA) will ease the extraction of relevant property information.

* Free software: BSD license

Documentation
-------------
Online: Checkout the documentation on GitLab_
Local: Run the following in the command line: ``python -m md_spa -d``

Installation
------------
* Step 1: Download the master branch from our gitlab page as a zip file, or clone it with git via ``git clone https://gitlab.nist.gov/gitlab/jac16/md-spa`` to your working directory.
* Step 2: Install with ``pip install md-spa/.``, or change directories and run ``pip install .``. Adding the flag ``-e`` will allow you to make changes that will be functional without reinstallation.

Features
--------

* Cluster Analysis (cluster)
* Coordination Number Analysis (coordination_number)
* Calculation of Volume with Monte Carlo (monte_carlo_volume)
* Radial Distribution Function (rdf)
* Read LAMMPS Files (read_lammps)
* Residence Time Analysis (residence_time)
* Intermediate Scattering Functions and Static Structure Factors (scattering)
* Viscosity (viscosity)
* Zeno output file handling (Zeno)

Credits
-------

This package was created with Cookiecutter_ and the `cookiecutter-nist-python`_ project template.

.. _GitLab: https://jac16.ipages.nist.gov/md-spa
.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`cookiecutter-nist-python`: https://gitlab.nist.gov/gitlab/jac16/cookiecutter-nist-python

