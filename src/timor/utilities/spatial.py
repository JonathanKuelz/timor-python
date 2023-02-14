#!/usr/bin/env python3
# Author: Jonathan Külz
# Date: 17.01.22
from typing import Collection

import hppfcl
import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation

import timor.utilities.errors as err

NO_TRANSLATION = np.zeros([3], float)
NO_ROTATION = np.eye(3)
NO_TRANSLATION.setflags(write=False)
NO_ROTATION.setflags(write=False)


# ----- Projections Start -----
def cartesian2cylindrical(cart: np.ndarray) -> np.ndarray:
    """
    Takes a (3,) array in cartesian coordinates [x, y, z] and returns the according cylindrical coordinates [r, phi, z].

    :source: https://en.wikipedia.org/wiki/Polar_coordinate_system
    """
    if not cart.squeeze().shape == (3,):
        raise ValueError("Expected (3,) array, got {}".format(cart.shape))
    x, y, z = cart
    r = (x ** 2 + y ** 2) ** .5
    if r == 0:
        phi = 0
    elif y >= 0:
        phi = np.arccos(x / r)
    else:
        phi = -np.arccos(x / r)
    return np.array([r, phi, z])


def cylindrical2cartesian(cyl: np.ndarray) -> np.ndarray:
    """Inverse to cartesian2cylindrical"""
    if not cyl.squeeze().shape == (3,):
        raise ValueError("Expected (3,) array, got {}".format(cyl.shape))
    r, phi, z = cyl
    return np.array([r * np.cos(phi), r * np.sin(phi), z], float)


def cartesian2spherical(cart: np.ndarray) -> np.ndarray:
    """
    Takes a (3,) array in cartesian coordinates [x, y, z] and returns the according spherical coordinates.

    Convention: [r, theta, phi]; r, theta, phi ~ radial, azimuthal, polar.
    :source: https://en.wikipedia.org/wiki/Spherical_coordinate_system
    """
    if not cart.squeeze().shape == (3,):
        raise ValueError("Expected (3,) array, got {}".format(cart.shape))
    x, y, z = cart
    r = (x ** 2 + y ** 2 + z ** 2) ** .5
    if r == 0:
        return np.array([0, 0, 0], float)

    theta = np.arccos(z / r)
    if x == 0:
        if y == 0:
            phi = 0
        elif y > 0:
            phi = np.pi / 2
        else:
            phi = -np.pi / 2
    elif x > 0:
        phi = np.arctan(y / x)
    else:
        if y >= 0:
            phi = np.arctan(y / x) + np.pi
        else:
            phi = np.arctan(y / x) - np.pi
    return np.array([r, theta, phi], float)


def spherical2cartesian(spher: np.ndarray) -> np.ndarray:
    """Inverse to cartesian2spherical"""
    if not spher.squeeze().shape == (3,):
        raise ValueError("Expected (3,) array, got {}".format(spher.shape))
    r, theta, phi = spher
    return np.array([r * np.cos(phi) * np.sin(theta), r * np.sin(phi) * np.sin(theta), r * np.cos(theta)], float)


def rot_mat2axis_angle(rot_mat: np.ndarray) -> np.ndarray:
    r"""
    Takes a rotation matrix and returns the corresponding axis-angle representation

    :param rot_mat: 3x3 rotation matrix
    :return: 4x1 array of axis-angle representation (n_x, n_y, n_z, theta_R), where n_x, n_y, n_z ~ unit vector and
        :math:`\theta_R` ~ rotation angle :math:`\in [0, \pi]`
    """
    axis_angle = Rotation.from_matrix(rot_mat.copy()).as_rotvec()
    if all(axis_angle == 0.):
        return np.zeros(4, float)
    axis_angle_4d = np.concatenate((axis_angle / np.linalg.norm(axis_angle),
                                    np.array([(np.pi + np.linalg.norm(axis_angle)) % (2 * np.pi) - np.pi])))
    if axis_angle_4d[3] < 0:
        # This can happen due to numerical inaccuracies, but only if the angle is almost equal to pi.
        theta = axis_angle_4d[3]
        if abs(theta + np.pi) > 1e-9:
            # This should never happen, thata is smaller zero but the "true" value is not pi.
            raise ValueError("Unexpected axis angle: {}".format(axis_angle_4d))
        axis_angle_4d[3] = np.pi
    return axis_angle_4d


def axis_angle2rot_mat(axis_angle: np.ndarray) -> np.ndarray:
    """
    Inverse to rot_mat2axis_angle.
    """
    rot_vec = axis_angle[:3] * axis_angle[3]
    return Rotation.from_rotvec(rot_vec).as_matrix()  # Calling copy to being able working with read-only arrays


# ----- Projections End -----


def axis_angle_rotation(vec: np.ndarray, axis_angle: np.ndarray) -> np.ndarray:
    """
    Rotates a vector by a given axis-angle representation.

    Equivalent to spatial.axis_angle2rot_mat(axis_angle) @ vec, but computationally more efficient.
    :source: https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation
    """
    if vec.squeeze().shape == (3,):
        theta = np.linalg.norm(axis_angle)
        axis = axis_angle / theta
    else:
        theta = axis_angle[3]
        axis = axis_angle[:3]
    if theta == 0:
        return vec
    return np.cos(theta) * vec + np.sin(theta) * np.cross(axis, vec) + (1 - np.cos(theta)) * np.dot(axis, vec) * axis


def clone_collision_object(co: hppfcl.CollisionObject) -> hppfcl.CollisionObject:
    """Deep copy of a hppfcl collision object"""
    return hppfcl.CollisionObject(co.collisionGeometry().clone(), co.getTransform())


def dh_extended_to_homogeneous(a: float = 0, alpha: float = 0, n: float = 0, p: float = 0, delta: float = 0):
    """
    Transform extended dh parameters into a homogeneous transformation matrix

    Describes a transformation similar to dh parameters (from Althoff et al., Sci. Robotics, 2019, Fig. 2)
    More precisely the transformation from a module input frame to the joint

    :param a: translate along x'_i [m]
    :param alpha: rotate around x_i [rad]
    :param delta: rotate around z_im1 [rad], gamma in Fig. 2
    :param p: offset of PJ_im1 to o'_i in z'_i direction [m]
    :param n: offset along z_i from o_i to PJ_i [m]
    """
    return homogeneous((0, 0, -p)) @ rotZ(-delta) @ homogeneous((a, 0, 0)) @ rotX(alpha) @ homogeneous((0, 0, n))


def euler2mat(rotation: Collection[float], seq: str):
    """
    Wrapper for the scipy euler method

    :param rotation: Rotation angles in radian
    :param seq: Any ordering of {xyz}[intrinsic] or {XYZ}[extrinsic] axes.
    :return: 3x3 rotation matrix
    """
    return Rotation.from_euler(seq, rotation).as_matrix()  # Calling copy to being able working with read-only arrays


def frame2geom(frame_id: int, geom_model: pin.GeometryModel):
    """Returns all geometry objects where porent frame is frame_id"""
    return [geom for geom in geom_model.geometryObjects if geom.parentFrame == frame_id]


def homogeneous(translation: Collection[float] = NO_TRANSLATION,
                rotation: Collection[Collection[float]] = NO_ROTATION) -> np.ndarray:
    """
    Returns a transformation matrix for translation, then rotation

    :param translation: Translation in the transformation matrix. 3x1
    :param rotation: 3x3 rotation matrix. Nested List works as well
    :return:
    """
    T = np.eye(4)
    T[:3, 3] = np.asarray(translation)
    T[:3, :3] = rotation
    return T


def inv_homogeneous(T: np.ndarray) -> np.ndarray:
    """Efficiently inverses a homogeneous transformation"""
    if T.shape != (4, 4):
        raise err.UnexpectedSpatialShapeError("Homogeneous transformation must be of shape 4x4")
    return homogeneous(-np.transpose(T[:3, :3]) @ T[:3, 3], np.transpose(T[:3, :3]))


def mat2euler(R: np.ndarray, seq: str = 'xyz') -> np.ndarray:
    """
    Inverse to euler2mat

    :param R: A 3x3 rotation matrix
    :param seq: Any ordering of {xyz}[intrinsic] or {XYZ}[extrinsic] axes. Defaults to roll-pitch-yaw
    :return: The rotations around the axes specified in seq that lead to R
    """
    if seq == 'xyz':
        shortcuts = (
            (NO_ROTATION, np.zeros(3)),
            (np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]), np.array([np.pi, 0, 0])),
        )
        for shortcut, angles in shortcuts:  # This saves a lot of time for the common cases
            if (R == shortcut).all():
                return angles
    return Rotation.from_matrix(R.copy()).as_euler(seq)  # Calling copy to being able working with read-only arrays


def random_rotation() -> np.ndarray:
    """Returns a random 3x3 rotation matrix."""
    return Rotation.random().as_matrix()


def rotation_in_bounds(rot: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """Maps angles to an interval 2pi-agnostically.

    Takes an array of rotations in radian and maps them to bounds aka limits if possible by adding/subtracting multiples
    of 2 pi. If not possible, will return the original input.

    :param rot: A 1xn array of rotations, given in radian.
    :param bounds: A 2xn array of limits, lower limits in first, upper limits in second row of the array. Every column
        in bounds corresponds to a column in rot.
    :returns: A variant of rot, offset by multiples of 2pi to fit within bounds. If that is not possible, returns rot.
    """
    if not bounds.shape == (2, rot.size):
        raise ValueError(f"Got rotation of shape {rot.shape} - would expect bounds of shape {(2, rot.size)}.")
    fitted = rot
    lower, upper = bounds[0, :], bounds[1, :]
    if not (lower <= upper).all():
        raise ValueError("Lower limits seem to bee greater than upper limits.")

    while (fitted < lower).any():
        fitted[fitted < lower] += 2 * np.pi
    while (fitted > upper).any():
        fitted[fitted > upper] -= 2 * np.pi
    if (lower <= fitted).all() and (upper >= fitted).all():
        return fitted
    return rot


def rot2D(angle: float) -> np.ndarray:
    """
    Returns the 2D rotation matrix for an angle in radian.

    :param angle: Angel in radian
    :return: 2D rotation
    """
    return np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])


def rotX(alpha: float) -> np.ndarray:
    """Returns a transformation rotation matrix for a rotation of a vector around the x-axis"""
    R = np.eye(4)
    R[1:3, 1:3] = rot2D(alpha)
    return R


def rotY(beta: float) -> np.ndarray:
    """Returns a transformation rotation matrix for a rotation of a vector around the y-axis"""
    R = np.eye(4)
    R[::2, ::2] = rot2D(beta).T
    return R


def rotZ(gamma: float) -> np.ndarray:
    """Returns a transformation rotation matrix for a rotation of a vector around the z-axis"""
    R = np.eye(4)
    R[:2, :2] = rot2D(gamma)
    return R


def skew(vec3: np.ndarray) -> np.ndarray:
    """Returns the skew symmetric matrix of a 3D vector"""
    if vec3.shape != (3,):
        raise err.UnexpectedSpatialShapeError("Invalid shape for skew symmetric matrix")
    return np.array([
        [0, -vec3[2], vec3[1]],
        [vec3[2], 0, -vec3[0]],
        [-vec3[1], vec3[0], 0]
    ])
