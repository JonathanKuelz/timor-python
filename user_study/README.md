# Where is the code for the user study?
In the paper "Timor Python: A Toolbox for Industrial Modular Robotics" presented at the IEEE/RSJ International Conference on Intelligent Robots and Systems 2023, Detroit, we showcase the functionalities of the toolbox by performing an optimization over Module Assemblies for a specified task.
The code to reproduce the results as well as the instructions given to the users are no longer part of the main branch of Timor.
Instead, the functionalities have been further developed and are fully included in the toolbox, mostly as part of the `timor.task.GA` module.

If you are interested in reproducing results from the paper or running the user study for yourself, you can clone the state of the repository at the time of submission of the paper. The commit hash is `c15ec613a439e0b259ac21efe7582eec06c4aae3`. You can also follow [this link](https://gitlab.lrz.de/tum-cps/timor-python/-/tree/c15ec613a439e0b259ac21efe7582eec06c4aae3) or search for the Gitlab tag `IROS_user_study`.

If you are interested in performing optimization of module assemblies yourself, we recommend starting with the `timor.task.GA` module.
