.. image:: https://gitlab.lrz.de/tum-cps/timor-python/-/raw/main/img/timor_banner.png
    :alt: Timor Banner
    :align: center
    :target: https://gitlab.lrz.de/tum-cps/timor-python/-/raw/main/img/timor_banner.png

Timor Python
============
The Toolbox for Industrial Modular Robot Design (Timor) is a python library for model generation and simulation of modular robots.

.. image:: https://gitlab.lrz.de/tum-cps/timor-python/badges/main/pipeline.svg
    :target: https://gitlab.lrz.de/tum-cps/timor-python/-/commits/%main
    :alt: Pipeline Status

.. image:: https://readthedocs.org/projects/timor-python/badge/?version=latest
    :target: https://timor-python.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. image:: https://gitlab.lrz.de/tum-cps/timor-python/badges/main/coverage.svg
    :target: https://gitlab.lrz.de/tum-cps/timor-python/-/jobs/artifacts/main/file/ci/coverage/html//index.html?job=coverage&min_acceptable=80&min_good=90
    :alt: Coverage Report

With Timor you can:

- Create, export and load sets of robot modules
- Assemble modules and generate kinematic, dynamic and collision models for you assembly
- Transfer your assembly to a `pinocchio <https://github.com/stack-of-tasks/pinocchio>`_-based robot model and perform forwards- and inverse kinematics and dynamics calculations
- Visualize modules and assemblies and animate trajectories
- Define robot tasks and evaluate solutions based on various cost functions
- Get started working with modular robots!

.. image:: https://gitlab.lrz.de/tum-cps/timor-python/-/raw/main/img/animations/demo_animation.gif
    :alt: Animation of a robot assembly moving between two goals
    :align: center
    :target: https://gitlab.lrz.de/tum-cps/timor-python/-/raw/main/img/animations/demo_animation.gif


Installation
------------
Timor-python is available on PyPI. It requires **at least Python 3.8**. For installation, use::

   pip install timor-python

Some requirements are not included by default - in order to install them, use::

  pip install timor-python[option]

where option can be one or multiple (comma-separated) of the following:

- ``dev``: Installs development requirements for local unittesting
- ``full``: Installs dev and jupyterlab -- needed to run the tutorials

If you want to work with the bleeding-edge version, you can download the source code from the project repository and install it locally.
Nagivate to the timor-python repository you cloned and enter::

   pip install -e .

to install it in editeable mode. This requires :code:`setuptools>=61` and :code:`pip>=21.3` (previous versions of setuptools require a :code:`setup.py`-file). To install optional dependencies, proceed in the same manner as for PyPI installs.

To run unittest locally, you need to `setup git lfs <https://git-lfs.com/>`_ in order to download assets.

If you want to use pre-commit hooks provided with Timor, for installation please use::

   pip install pre-commit

then::

   pre-commit install


After that, each time you commit files, it will automatically perform linting and style checks.

Usage
-----
The `tutorials folder <tutorials/>`_ contains jupyter notebooks that cover the most common use cases for Timor.
To open and run the notebooks, you will need jupyter, which can be installed from PyPI::

  pip install jupyterlab

To inspect, run, or edit the tutorials, navigate to the tutorials folder and start the notebook::

  jupyter lab

You can set custom configurations such as file paths of robot libraries or logging behavior by editing the config file. You can import the file location of the config file as :code:`from timor.utilities.configurations import CONFIG_FILE`.

For further information, please visit the `documentation <https://timor-python.readthedocs.io>`_.

Updating
--------

Some timor updates might require to re-acquire the task, schema, or robot module data-bases. Try to reset the caches by running ``timor.utilities.file_locations.clean_caches()`` and re-import the timor module.

Typical errors indicating this action:
 * ValueError: modules.json invalid.

Support
-------
Do you have a question or an issue using Timor? You can either `submit an issue <https://gitlab.lrz.de/tum-cps/timor-python/-/issues>`_ or write an email to the `repository maintainer <mailto:jonathan.kuelz@tum.de>`_.

Contributing
------------
We welcome every contribution to Timor. For more details, please refer to our `contribution guidelines <https://gitlab.lrz.de/tum-cps/timor-python/-/blob/main/CONTRIBUTING.md>`_.

Authors and acknowledgment
--------------------------
Timor was developed at the chair of robotics, artificial intelligence and embedded systems at TU Munich.
It is designed, developed and maintained by Jonathan Külz, Matthias Mayer, and Matthias Althoff.

The Timor Python logo was AI-generated using the OpenAI's Dall-E 2 API.

The developers gratefully acknowledge financial support by the Horizon 2020 EU Framework Project `CONCERT <https://concertproject.eu/>`_.
