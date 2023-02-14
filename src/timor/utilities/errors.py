#!/usr/bin/env python3
# Author: Jonathan Külz
# Date: 17.02.22
"""Custom error messages for robotics applications"""
import numpy as np
import numpy.testing as np_test


class DuplicateValueError(Exception):
    """
    This can be raised whenever a duplicate value is added to a container.

    Containers, such as a list, set, that allows unique values only.
    """


class InvalidAssemblyError(Exception):
    """This can be raised when an assembly of modules should be created that's not valid"""


class UniqueValueError(ValueError):
    """This error can be raised whenever a values, such as an ID or name should be unique, but isn't"""


class UnexpectedSpatialShapeError(ValueError):
    """Can be thrown whenever a spatial input (point, rotation, transformation) has the wrong input shape"""


class NonOrthogonalRotationError(ValueError):
    """Raised when a rotation matrix is not orthogonal"""


class RotationVolumeError(ValueError):
    """Raised when a rotation matrix is not volume preserving"""


class TimeNotFoundError(IndexError):
    """Raised when a time step t can not be found in a trajectory, solution, etc. that is indexed at this time t."""


def assert_has_3d_point(p: np.ndarray):
    """
    Checks whether the input can be interpreted as a point in cartesian space

    :raises: UnexpectedSpatialShapeError
    """
    s = p.shape
    if s not in [(3,), (1, 3), (3, 1), (4, 4)]:
        raise UnexpectedSpatialShapeError(f"Array of shape {s} cannot be interpreted as 3D point.")


def assert_is_3d_point(p: np.ndarray):
    """
    Stronger assertion on points in cartesian space

    :raises: UnexpectedSpatialShapeError
    """
    s = p.shape
    if s not in [(3,), (3, 1)]:
        raise UnexpectedSpatialShapeError(f"Array of shape {s} is not a point in 3D space.")


def assert_is_rotation_matrix(R: np.ndarray):
    """
    Checks whether a 3x3 matrix is a valid rotation (orthonormal)

    :raises: AssertionError
    """
    if R.shape != (3, 3):
        raise UnexpectedSpatialShapeError(f"Rotation matrix must be of shape (3, 3). Given: {R.shape}")
    if not np.allclose(np.eye(3), R @ R.T, atol=1e-9):
        raise NonOrthogonalRotationError("Rotation matrix is not orthogonal")
    if not np.allclose(np.linalg.det(R), 1, atol=1e-9):
        raise RotationVolumeError("Rotation matrix is not volume preserving")


def assert_is_homogeneous_transformation(T: np.ndarray):
    """
    Checks for shape, valid rotation matrix and last row

    :raises: UnexpectedSpatialShapeError, AssertionError
    """
    if T.shape != (4, 4):
        raise UnexpectedSpatialShapeError(
            f"Array of shape {T.shape} cannot be interpreted as homogeneous transformation.")
    assert_is_rotation_matrix(T[:3, :3])
    np_test.assert_allclose(T[3, :], np.array([.0, .0, .0, 1.]), atol=1e-9)
