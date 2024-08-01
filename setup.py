#!python
from pathlib import Path
import sys

import cmeel.config
from pybind11.setup_helpers import Pybind11Extension
from setuptools import setup


cmeel_base_path = Path(cmeel.__file__).parent.parent.joinpath(Path(cmeel.config.CMEEL_PREFIX))
if not cmeel_base_path.joinpath('include').joinpath('eigen3').is_dir():
    print(f"Could not find eigen3 directory in {cmeel_base_path.joinpath('include')}")
    print("Possibly need to run `pip install cmeel-eigen`")

ext_modules = [
    Pybind11Extension(
        "timor.utilities.ik_cpp",
        sorted(("src/timor/utilities/ik_cpp.cpp",)),
        include_dirs=[str(cmeel_base_path.joinpath('include').joinpath('eigen3')),
                      str(cmeel_base_path.joinpath('include'))],
        cxx_std=14,
        libraries=[f'boost_python{sys.version_info.major}{sys.version_info.minor}',
                   'hpp-fcl'],  # For collision checking
        library_dirs=[str(cmeel_base_path.joinpath('lib'))],
        runtime_library_dirs=[str(cmeel_base_path.joinpath('lib'))],
        extra_compile_args=['-O2',
                            '-DPINOCCHIO_WITH_HPP_FCL=TRUE',  # For collision checking
                            ]
    ),
]

if __name__ == '__main__':
    setup(ext_modules=ext_modules)
