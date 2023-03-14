This folder contains the code used to conduct the user study as described in the accompanying paper to this toolbox.
As the user study itself is not part of the toolbox functionalities, it is not included in the testing suite and forwards compatability is not guaranteed.
The scripts in this folder have last been tested on the commit with hash xxx - if you check out this hash from the project repository, compatability with the local code is ensured.

# Evaluate your robot building skills

Welcome to the timor user study.
Here, you can see how well you perform in building a modular robot, compared to a brute force algorithm.
You have 30 minutes of time, which is a bit more than a Desktop PC needs to run the implemented brute force approach.

For instructions, have a look at the `user study instruction document <user_study_instructions.pdf/>`_.
You will also need the additionally provided `module overview <user_study_modules.pdf/>`_.

**To get started** just install timor-python (on a Linux machine), cd to this directory and run::

    python user_study.py

In order to run the brute force algorithm, you need to :code:`pip install tqdm` first, then you can run::

  python brute_force_solution.py

Have fun!
