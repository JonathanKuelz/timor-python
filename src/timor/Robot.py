#!/usr/bin/env python3
# Author: Jonathan Külz
# Date: 12.01.22
from __future__ import annotations

import abc
import itertools
from pathlib import Path
import string
from typing import Callable, Collection, Dict, List, Optional, Tuple, Union
from warnings import warn

from hppfcl.hppfcl import CollisionObject, CollisionRequest, CollisionResult, Transform3f, collide
from matplotlib import colors
import numpy as np
import pinocchio as pin
import scipy.optimize as optimize

from timor.task.Obstacle import Obstacle
from timor.utilities import logging, spatial
from timor.utilities.dtypes import IntermediateIkResult, float2array
from timor.utilities.tolerated_pose import ToleratedPose
from timor.utilities.transformation import Transformation, TransformationLike


class PinBody:
    """A wrapper for a body in pinocchio, which is usually represented only implicitly"""

    def __init__(self, inertia: pin.Inertia, placement: pin.SE3, name: str,
                 collision_geometries: List[pin.GeometryObject] = None,
                 visual_geometries: List[pin.GeometryObject] = None
                 ):
        """
        Set up a container for a pinocchio body.

        :param inertia: the inertia of the body, as a pinocchio inertia object
        :param placement: the placement of the body, as a pinocchio SE3 object
        :param name: the name of the body, should be unique within the robot
        :param collision_geometries: the collision geometries of the body
        :param visual_geometries: the visual geometries of the body
        """
        self.inertia: pin.Inertia = inertia
        self.placement: pin.SE3 = placement
        self.name: str = name
        self.collision_geometries: List[pin.GeometryObject] = collision_geometries if collision_geometries else []
        self.visual_geometries: List[pin.GeometryObject] = visual_geometries if visual_geometries else []

    def body_data(self, parent_joint_id: int):
        """Returns a dictionary that can be used as kwargs for appendBodyToJoint (pinocchio)"""
        return {'joint_id': parent_joint_id, 'body_inertia': self.inertia, 'body_placement': self.placement}

    def body_frame_data(self, parent_joint_id: int, parent_frame_id: int):
        """Returns a dictionary that can be used as kwargs for addBodyFrame (pinocchio)"""
        return {'body_name': self.name,
                'parentJoint': parent_joint_id,
                'body_placement': self.placement,
                'previous_frame': parent_frame_id}


def default_ik_cost_function(robot: RobotBase, q: np.ndarray, goal: Transformation) -> float:
    """
    The default cost function to evaluate the quality of an inverse kinematic solution.

    This method returns the weighted sum of the translation and rotation error between the current and the goal, where
    the 1m of displacement equals a cost of one and 180 degree of orientation error equals a cost of 1.
    :param robot: The robot to evaluate the solution for
    :param q: The current joint configuration
    :param goal: The goal transformation
    """
    current = robot.fk(q, kind='tcp')
    delta = current.distance(goal)
    translation_error = delta.translation_euclidean
    rotation_error = delta.rotation_angle
    translation_weight = 1.
    rotation_weight = .5 / np.pi  # .5 meter displacement ~ 180 degree orientation error
    return (translation_weight * translation_error + rotation_weight * rotation_error) / \
        (translation_weight + rotation_weight)


class RobotBase(abc.ABC):
    """Abstract base class for all robot classes"""

    def __init__(self,
                 *,
                 name: str,
                 home_configuration: np.ndarray = None,
                 base_placement: Optional[TransformationLike] = None
                 ):
        """
        RobotBase Constructor

        :param home_configuration: Home configuration of the robot (default pin.neutral)
        :param base_placement: Home configuration of the robot (default pin.neutral)
        """
        self._name: str = name
        if home_configuration is None:
            home_configuration = np.zeros((self.njoints,))
        self.configuration: np.ndarray = np.empty_like(home_configuration)
        self.update_configuration(home_configuration)
        self.velocities: np.ndarray = np.zeros(self.configuration.shape)
        self.accelerations: np.ndarray = np.zeros(self.configuration.shape)

        if base_placement is not None:
            self._base_placement = Transformation(base_placement)
            self.move(base_placement)
        else:
            self._base_placement = Transformation.neutral()

    # ---------- Static- and Classmethods ----------
    @classmethod
    @abc.abstractmethod
    def from_urdf(cls, urdf_file: Path, package_dir: Path, **kwargs) -> RobotBase:
        """Class constructor from urdf file. kwargs should at least handle RobotBase keyword arguments."""

    # ---------- Abstract Properties ----------
    @property
    @abc.abstractmethod
    def joints(self) -> List[str]:
        """Returns a list containing all joint IDs of the robot"""

    @property
    @abc.abstractmethod
    def mass(self) -> float:
        """Total robot mass in kilogram"""

    @property
    @abc.abstractmethod
    def njoints(self) -> int:
        """Returns the number of joints in the robot."""

    @property
    @abc.abstractmethod
    def parents(self) -> Dict[str, str]:
        """Returns a mapping between joints and their parents. The base/world joint is its own parent"""

    @property
    @abc.abstractmethod
    def placement(self) -> Transformation:
        """The (base) placement in the world coordinate system (4x4 homogeneous transformation / placement)"""

    @placement.setter
    @abc.abstractmethod
    def placement(self, placement: TransformationLike):
        """Define the (base) placement of the robot in the world coordinate system (4x4 homogeneous transformation)"""

    @property
    @abc.abstractmethod
    def joint_limits(self) -> np.ndarray:
        """Returns an array of shape 2xnjoints containing the lower and upper joint position limits"""

    @property
    @abc.abstractmethod
    def joint_acceleration_limits(self) -> np.ndarray:
        """Returns an array of shape 1xnjoints containing the absolute joint acceleration limits"""

    @property
    @abc.abstractmethod
    def joint_torque_limits(self) -> np.ndarray:
        """Returns an array of shape 1xnjoints containing the absolute joint torque limits"""

    @property
    @abc.abstractmethod
    def joint_velocity_limits(self) -> np.ndarray:
        """Returns an array of shape 1xnjoints containing the absolute joint velocity limits"""

    @property
    @abc.abstractmethod
    def tcp_velocity(self) -> np.ndarray:
        """
        Returns the current tcp velocity as 6x1 numpy array
        """

    @property
    @abc.abstractmethod
    def tcp_acceleration(self) -> np.ndarray:
        """
        Returns the current tcp acceleration as 6x1 numpy array
        """

    # ---------- Abstract Methods ----------
    @abc.abstractmethod
    def collisions(self, task: 'Task.Task') -> List[Tuple[RobotBase, Obstacle]]:  # noqa: F821
        """
        Returns all collisions that appear between the robot and itself or obstacles in the task.

        :param task: The task to check for collisions
        :return: A list of tuples (robot, obstacle) for robot an obstacle in collision
        """

    @abc.abstractmethod
    def fk(self, configuration: np.array = None, kind: str = 'tcp') -> Union[Transformation, Tuple[Transformation]]:
        """
        Returns the forward kinematics for the robot as homogeneous transformation.

        :param configuration: Joint configuration the fk should be given for. If None, will take current configuration.
        :param kind: Depends on the class, but should at least support 'tcp' and 'joints'.
        """

    @abc.abstractmethod
    def has_collisions(self, task: 'Task.Task') -> bool:  # noqa: F821
        """
        Returns true if there is at least one collision in the task or one self-collision.

        Faster than self.collisions in case of collision, as this method breaks as soon as the first one is detected.

        :param task: The task the robot is in.
        :return: Boolean indicator whether there are any collisions.
        """

    @abc.abstractmethod
    def has_self_collision(self, q: np.ndarray = None) -> bool:
        """
        Returns true if there are any self collisions in configuration q (defaults to current configuration)
        """

    @abc.abstractmethod
    def id(self, q: np.ndarray = None, dq: np.ndarray = None, ddq: np.ndarray = None, **kwargs) -> np.ndarray:
        """
        Inverse Dynamics solver, expected to return torques as 1xnJoints array.

        Should work with current configuration if not explicitly given q, dq and ddq.
        """

    @abc.abstractmethod
    def move(self, displacement: TransformationLike) -> None:
        """
        Moves the whole robot by displacement

        :param displacement: A 4x4 Transformation
        """

    @abc.abstractmethod
    def random_configuration(self) -> np.ndarray:
        """
        Returns a random configuration for the robot
        """

    @abc.abstractmethod
    def set_base_placement(self, placement: TransformationLike) -> None:
        """
        Places the robot base at placement.

        :param placement: 4x4 Transformation
        """

    @abc.abstractmethod
    def static_torque(self, q: np.ndarray, f_ext=None) -> np.ndarray:
        """
        Computes the static torques in the robot joints composed of g(q) and tau_ext

        :param q: Current joint position
        :param f_ext: External forces
        :return: n_joints x 1 array of torques
        """

    @abc.abstractmethod
    def update_configuration(self, q: np.ndarray, dq: np.ndarray = None, ddq: np.ndarray = None) -> None:
        """
        Updates the current joint angles/configuration.

        :param q: Dimension njointsx1.
        :param dq: Velocities, optional
        :param ddq: Acceleration, optional
        """

    @abc.abstractmethod
    def manipulability_index(self, q: Optional[np.ndarray]) -> float:
        """
        Calculate the Yoshikawa manipulability index at joint position q or current one.

        :param q: Joint position to calculate at (default current robot configuration)
        """

    @abc.abstractmethod
    def visualize(self, viz: pin.visualize.MeshcatVisualizer, *args, **kwargs) -> pin.visualize.MeshcatVisualizer:
        """Interface to plot the robot in a meshcat visualizer"""

    # ---------- Properties and Aliases----------

    @property
    def name(self) -> str:
        """Unique robot name"""
        return self._name

    @property
    def q(self) -> np.ndarray:
        """Alias for self.configuration"""
        return self.configuration

    @property
    def dq(self) -> np.ndarray:
        """Alias for self.velocities"""
        return self.velocities

    @property
    def ddq(self) -> np.ndarray:
        """Alias for self.accelerations"""
        return self.accelerations

    # ---------- Methods ----------

    def collision_info(self, task: 'Task.Task'):  # noqa: F821
        """
        Calculates all collisions in the task and prints human-readable information.

        :param task: Task to test for collisions
        """
        print(f"{task.id} contains the following collisions:")
        i = -1
        for i, (robot, obstacle) in enumerate(self.collisions(task)):
            print(f"{robot.name} collides with {obstacle.display_name}.")
        if i >= 0:
            print(f"Total: {i + 1}")
        else:
            print("No collisions")

    def ik(self,
           eef_pose: ToleratedPose,
           q_init: np.ndarray = None,
           **kwargs) -> Tuple[np.ndarray, bool]:
        """
        Calculate an inverse kinematics with this robot.

        :param eef_pose: The desired 4x4 placement of the end effector
        :param q_init: The joint configuration to start with. If not given, will start the iterative optimization at the
            current configuration
        :return: A tuple of (q_solution, success [boolean])
        """
        return self.ik_scipy(eef_pose, q_init, **kwargs)

    def ik_scipy(self,
                 eef_pose: ToleratedPose,
                 q_init: np.ndarray = None,
                 max_iter: int = 1000,
                 restore_config: bool = False,
                 max_n_tries: int = 10,
                 ik_cost_function: Callable[[RobotBase, np.ndarray, TransformationLike], float] = default_ik_cost_function,  # noqa: E501
                 **kwargs
                 ) -> Tuple[np.ndarray, bool]:
        """
        Computes the inverse kinematics (joint angles for a given goal destination).

        :param eef_pose: Desired end effector pose with tolerances.
        :param q_init: Initial joint angles to start searching from. If not given, uses current configuration
        :param max_iter: Maximum number of iterations before the algorithm gives up. If max_n_tries is larger than 1,
            max_iter is counted over all n tries.
        :param restore_config: If True, the current robot configuration will be kept. If false, the found solution for
            the inverse kinematics will be kept
        :param max_n_tries: Maximum number of runs with a new random init configuration before the algorithm gives up
        :param ik_cost_function: A custom cost function that takes a robot, its joint angles, and a
            TransformationLike and returns a scalar cost indicating the quality of the configuration in solving the ik
            problem. If not given, the default cost function is used (equal weighting of .5 meter / 180 degree error).
        :param kwargs: Additional arguments, including:
            - task: If a timor.Task.Task is provided here, the ik will restart if there are task collisions as long as
                max_iter is not reached.
            - joint_mask: A boolean array of length njoints that indicates which joints should be considered for the IK
            - allow_random_restart: The IK will restart from random configurations if it fails to converge. This can be
                disabled by setting this to False.
            - ignore_self_collision: If True, the IK will ignore self-collisions
            - method: The scipy minimize method to be used. Defaults to 'L-BFGS-B'
            - ftol: The tolerance for the cost function, i.e. the algorithm will stop if the cost function changes less
                than this amount between iterations. Defaults to 1e-8.
        :return: A tuple of q_solution [np.ndarray], success [boolean]. If success is False, q_solution is the
            configuration with minimal IK cost amongst all seen in between iterations if this method.
        """
        if 'tolerance' in kwargs:
            raise ValueError("The pose needs to be tolerated - providing an extra tolerance is deprecated.")

        if 'joint_mask' in kwargs:
            if q_init is None:
                raise ValueError("If you provide a joint mask, you need to provide a q_init as well.")
            joint_mask = np.asarray(kwargs['joint_mask']).astype(bool)
        else:
            joint_mask = np.ones_like(self.configuration, dtype=bool)

        if self.njoints == 0:
            return self.configuration, eef_pose.valid(self.fk())

        previous_config = self.configuration
        if q_init is None:
            q_start = self.random_configuration()
        else:
            q_start = q_init

        q_masked = q_start[joint_mask]

        def _unmask(q_partial):
            q_ik = q_start
            q_ik[joint_mask] = q_partial
            return q_ik

        def _cost(_q: np.ndarray):
            return ik_cost_function(self, _unmask(_q), eef_pose.nominal)

        lowest_cost = kwargs.get('lowest_cost', IntermediateIkResult(_unmask(q_masked), _cost(q_masked)))

        bounds = optimize.Bounds(self.joint_limits[0, joint_mask], self.joint_limits[1, joint_mask])
        sol = optimize.minimize(
            _cost,
            q_masked,
            method=kwargs.get('method', 'L-BFGS-B'),
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': kwargs.get('ftol', 1e-8)}
        )

        q_sol = _unmask(sol.x)
        if sol.fun < lowest_cost.distance:
            lowest_cost = IntermediateIkResult(q_sol, sol.fun)

        if restore_config:
            self.update_configuration(previous_config)  # Restore the old configuration

        # sol.success gives information about successfull termination, but we are not interested in that
        success = eef_pose.valid(self.fk(q_sol))

        if success and self.has_self_collision() and not kwargs.get('ignore_self_collision', False):
            success = False
        elif success and 'task' in kwargs and self.has_collisions(kwargs['task']):
            success = False
        elif success and kwargs.get('check_static_torques', False):
            tau = self.static_torque(q_sol)
            success = (np.abs(tau) <= self.joint_torque_limits).all()

        logging.debug("Scipy IK ends with message: %s", sol.message)
        if not success and max_n_tries > 0:
            logging.debug("Scipy IK failed, trying again with new random init configuration")
            kwargs['lowest_cost'] = lowest_cost
            return self.ik_scipy(eef_pose, q_init, max_iter - sol.nit, restore_config, max_n_tries - 1, **kwargs)

        if success or 'lowest_cost' not in kwargs:
            q = q_sol
        else:
            logging.debug("Scipy IK failed, using lowest cost solution from all tries")
            q = lowest_cost.q
        return q, success

    def q_in_joint_limits(self, q: np.ndarray) -> bool:
        """Returns true if a configuration q is within the joint limits of the robot"""
        above_lower = np.all(self.joint_limits[0, :] <= q)
        below_upper = np.all(q <= self.joint_limits[1, :])
        return above_lower and below_upper


class PinRobot(RobotBase):
    """
    An implementation of the robot base class, using the pinocchio library.

    `Pinocchio <https://github.com/stack-of-tasks/pinocchio>`_ from stack of tasks (pinocchio-python) for modelling
    kinematics and dynamics.
    """

    def __init__(self,
                 *,
                 wrapper: pin.RobotWrapper,
                 home_configuration: np.ndarray = None,
                 base_placement: TransformationLike = None
                 ):
        """
        PinRobot Constructor

        :param wrapper: Pinocchio robot wrapper.
        """
        if wrapper.collision_model is None:
            wrapper.collision_model = pin.GeometryModel()
        if wrapper.visual_model is None:
            wrapper.visual_model = pin.GeometryModel()

        self.model: pin.Model = wrapper.model
        self.data: pin.Data = wrapper.data
        self.collision: pin.GeometryModel = wrapper.collision_model
        self.visual: pin.GeometryModel = wrapper.visual_model

        super().__init__(name=self.model.name,
                         home_configuration=home_configuration,
                         base_placement=None)

        self._init_collision_pairs()
        self.collision_data: pin.GeometryData = self.collision.createData()
        self.visual_data: pin.GeometryData = self.visual.createData()
        self._update_geometry_placement()
        self._remove_home_collisions()

        if base_placement is not None:
            # This needs to be done after initializing the geometry objects, so outside the super constructor
            self.move(base_placement)

        self._tcp = self.model.nframes - 1

    def __deepcopy__(self, memodict={}):
        """Pin Robot cannot be deep-copied without losing the information on how bodies are connected."""
        raise NotImplementedError("Deep Copy for pinnocchio robot currently not supported!")

    def __getstate__(self):
        """
        Custom getstate (used for pickling) for pinocchio robot in order to enable multiprocessing.

        Geometric data from pinocchio is no serializeable
        """
        warn("PICKLE OF ROBOT CLASS DOES EXCLUDE GEOMETRY DATA")
        d = self.__dict__.copy()
        del d['collision']
        del d['visual']
        del d['collision_data']
        del d['visual_data']
        return d

    def __setstate__(self, state):
        """Unpickle."""
        self.__dict__ = state

    # ---------- Static- and Classmethods ----------
    @classmethod
    def from_urdf(cls, urdf_file: Path, package_dir: Path, **kwargs) -> PinRobot:
        """Utility wrapper to load information about a robot from URDF and build the according PinRobot."""
        wrapper = pin.RobotWrapper.BuildFromURDF(str(urdf_file), str(package_dir))
        kwargs.setdefault('home_configuration', pin.neutral(wrapper.model))
        return cls(
            wrapper=wrapper,
            **kwargs
        )

    # ---------- Properties ----------
    @property
    def collision_pairs(self) -> np.ndarray:
        """
        Uses the collision_objects property and returns all pairs that are in collision

        :return: An nx2 numpy array of collision pair inidices, where n is the number of pairs in collision
        """
        cp = np.asarray(
            [[cp.first, cp.second] for cp in self.collision.collisionPairs],
            dtype=int
        )
        if cp.size == 0:
            cp = np.empty([0, 2])
        return cp

    @property
    def collision_objects(self) -> Tuple[CollisionObject]:
        """
        Returns all collision objects of the robot (internally wrapped by pinocchio) as hppfcl collision objects.

        :returns: A list of hppfcl collision objects that can be used for fast collision checking.
        """
        return tuple(self.collision.geometryObjects)

    @property
    def joints(self) -> List[str]:
        """Returns the names of the "real" joints in the robot. (ignoring universe, which is, in pinocchio, a joint)"""
        return list(self.model.names)[1:]  # The universe is not really a joint

    @property
    def mass(self) -> float:
        """
        Robot mass in kg, included the unactuated mass belonging to the base.

        Pinocchio calculates the robot mass as the actuated mass only, omitting the mass attached to the "universe".
        Opposed to this approach, we include the mass of the (static) base in the robot mass.
        :source: https://github.com/stack-of-tasks/pinocchio/issues/1286
        """
        return pin.computeTotalMass(self.model) + self.model.inertias[0].mass

    @property
    def njoints(self) -> int:
        """Number of actuated joints."""
        return self.model.njoints - 1  # The universe is not really a joint

    @property
    def parents(self) -> Dict[str, str]:
        """Returns a mapping from a joint to its parent. Assumes there is a unique parent for each joint!"""
        return {
            jnt: self.model.names[self.model.parents[self.model.getJointId(jnt)]] for jnt in self.joints
        }

    @property
    def placement(self) -> Transformation:
        """The robot's base placement. Wrapped as property so it can only be set using the according methods."""
        return self._base_placement

    @placement.setter
    def placement(self, value: TransformationLike):
        raise NotImplementedError("Use move instead of setting the placement directly.")

    @property
    def joint_limits(self) -> np.ndarray:
        """Numpy array where arr[0,:] are the lower joint limits and arr[1,:] are the upper joint limits."""
        return np.vstack([self.model.lowerPositionLimit, self.model.upperPositionLimit])

    @property
    def joint_acceleration_limits(self) -> np.ndarray:
        """Numpy array where arr[0,:] are the lower joint limits and arr[1,:] are the upper acceleration limits."""
        warn("Acceleration Limits not Implemented for Robots! Set to infinity")
        return np.ones((self.njoints,), float) * np.inf

    @property
    def joint_torque_limits(self) -> np.ndarray:
        """Numpy array where arr[0,:] are the lower joint limits and arr[1,:] are the upper torque limits."""
        return self.model.effortLimit

    @property
    def joint_velocity_limits(self) -> np.ndarray:
        """Numpy array where arr[0,:] are the lower joint limits and arr[1,:] are the upper velocity limits."""
        return self.model.velocityLimit

    @property
    def tcp(self):
        """The index of the frame indicating the robot's TCP (Tool Center Point)."""
        return self._tcp

    @tcp.setter
    def tcp(self, frame_id):
        """
        Set the frame of the tool center point.

        :param frame_id: The frame id of the tool center point.
        """
        previous = self._tcp
        self._tcp = frame_id

        if (previous != self._tcp) and (self.model.frames[self._tcp] == 'TCP'):
            # TODO: Would be nice to remove the old TCP frame, but currently I don't see this option in pinocchio
            pass

    @property
    def tcp_parent(self) -> str:
        """Returns the NAME of the joint the tcp frame is attached to"""
        parent_internal_index = self.model.frames[self.tcp].parent
        return self.model.names[parent_internal_index]

    @property
    def tcp_velocity(self, reference_frame=pin.ReferenceFrame.LOCAL_WORLD_ALIGNED) -> np.ndarray:
        r"""
        Returns the current tcp velocity as 6x1 numpy array - per default relative to the world coordinate system.

        :param reference_frame: Select reference frame to represent tcp velocity in; details see:
           https://docs.ros.org/en/kinetic/api/pinocchio/html/group__pinocchio__multibody.html
        :return: Array math:`(\Delta x, \Delta y, \Delta z, \Delta \alpha, \Delta \beta, \Delta \gamma) / \Delta t`
        """
        return pin.getFrameVelocity(self.model, self.data, self.tcp, reference_frame).np

    @property
    def tcp_acceleration(self) -> np.ndarray:
        """
        Returns the current tcp velocity as 6x1 numpy array - relative to the world coordinate system.

        :return: Array (ddx, ddy, ddz, dd_alpha, dd_beta, dd_gamma) / dt
        """
        reference = pin.ReferenceFrame.WORLD
        return pin.getFrameAcceleration(self.model, self.data, self.tcp, reference).np

    # ---------- Utilities and Dynamic and Kinematics Methods ----------
    def _update_geometry_placement(self):
        """Updates the collision (for collision detection) and visual (for plotting) geometry placements."""
        self._update_collision_placement()
        self._update_visual_placement()

    def _update_kinematic(self):
        """Can be called after changing the robot structure. This method updates the kinematic model of the robot"""
        self._init_collision_pairs()
        self.data, self.collision_data, self.visual_data = pin.createDatas(self.model, self.collision, self.visual)
        self.update_configuration(self.configuration)  # Add new joint with q=0

    def friction(self, dq: np.ndarray) -> np.ndarray:
        """Computes the forces due to friction"""
        # sin(atan( )) is smooth signum function
        return self.model.damping * dq + self.model.friction * np.sin(np.arctan(100 * dq))

    def manipulability_index(self, q: Optional[np.ndarray] = None) -> float:
        """
        Calculate the Yoshikawa manipulability index at joint position q or current one.

        :param q: Joint position to calculate at (default current robot configuration)
        """
        if q is None:
            q = self.configuration

        J = pin.computeJointJacobian(self.model, self.data, q, self.model.frames[self.tcp].parent)
        return np.sqrt(np.linalg.det(J @ J.T))

    def motor_inertias(self, ddq: np.ndarray) -> np.ndarray:
        """Computes the motor inertias in the joins at a given acceleration"""
        return ddq * np.power(self.model.rotorGearRatio, 2) * self.model.rotorInertia

    def fk(self, configuration: np.array = None, kind: str = 'tcp', collision: bool = False, visual: bool = False):
        """
        Calculate the forward kinematics (joint state -> end-effector position).

        :param configuration: Joint configuration the fk should be given for. If None, will take current configuration.
        :param kind: Either 'tcp', 'joints' or 'full'.
        :param collision: Perform updates on the collision geometry objects placements
        :param visual: Perform updates on the visual geometry objects placements
        :return: np. array of either dim 4x4 (kind tcp) or numJointsx4x4 (kind joints) or numFramesx4x4 (kind full)
        """
        if configuration is not None:
            self.update_configuration(configuration)

        if kind == 'joints':
            # Assuming no one wants to have the "fixed universe joint" represented by the base
            fkin = np.asarray([trans.homogeneous for trans in self.data.oMi], float)[1:, :, :]
            fkin = tuple(Transformation(joint_pose) for joint_pose in fkin)
        elif kind == 'tcp':
            pin.updateFramePlacement(self.model, self.data, self.tcp)
            fkin = Transformation(self.data.oMf[self.tcp].homogeneous)
        elif kind == 'full':
            pin.updateFramePlacements(self.model, self.data)
            fkin = np.asarray([trans.homogeneous for trans in self.data.oMf], float)
            fkin = tuple(Transformation(frame_pose) for frame_pose in fkin)
        else:
            raise ValueError("Unknown kind {} for forward kinematics.".format(kind))

        if collision:
            self._update_collision_placement()
        if visual:
            self._update_visual_placement()

        return fkin

    def fd(self, tau: np.ndarray, q: np.ndarray = None, dq: np.ndarray = None,
           motor_inertia: bool = False, friction: bool = True) -> np.ndarray:
        """
        Solves the forward dynamics assuming no external wrench (= force + torque).

        By default considers friction and motor inertias. The inverse operation is the "id" method.

        :param tau: Torques commanded
        :param q: Joint position; defaults to current position (self.configuration)
        :param dq: Joint velocity; defaults to current velocity (self.velocities)
        :param motor_inertia: Consider motor inertias
        :param friction: Consider joint friction
        :return: ddq the resulting joint accelerations
        """
        q = self.configuration if q is None else q
        dq = self.velocities if dq is None else dq
        tau = tau - friction * self.friction(dq)
        if motor_inertia:
            raise NotImplementedError("Motor inertia not yet considered in Robot.fd")
        ddq = pin.aba(self.model, self.data, q, dq, tau)
        return ddq

    def id(self,
           q: np.ndarray = None,
           dq: np.ndarray = None,
           ddq: np.ndarray = None,
           motor_inertia: bool = True,
           friction: bool = True,
           eef_wrench: np.ndarray = None,
           eef_wrench_frame: str = "world") -> np.ndarray:
        """
        Solves inverse dynamics using recursive newton euler, assuming no external forces or torques.

        Solves the equation M * a_q + b = tau_q, where a_q is the joint acceleration (ddq), M is the mass matrix,
        b is the drift (composed of Coriolis, ???, and gravitation) and tau_q is the applied torques in the joints.

        :param motor_inertia: Whether to consider torques to balance motor side inertia
        :param friction: Whether to consider friction compensating torques
        :param eef_wrench: [f_x, f_y, f_z, tau_x, tau_y, tau_z] acting on the end-effector
        :param eef_wrench_frame: Frame the eef_wrench is described in (still acting on eef position!);
            for now only "world" accepted. TODO : Extend with future frame definitions
        """
        q = self.configuration if q is None else q
        dq = self.velocities if dq is None else dq
        ddq = self.accelerations if ddq is None else ddq
        known_frames = {"world"}  # TODO : Extend with future frame definitions
        if eef_wrench_frame not in known_frames:
            raise ValueError(f"{eef_wrench_frame} not in known frames {known_frames}")
        # pin expects (nDoF,) array but 1D robot can produce scalar q's
        f_ext_vec = pin.StdVec_Force()  # Force acting in each _joint_ frame
        for i in range(self.njoints):
            f = pin.Force()
            f.setZero()
            f_ext_vec.append(f)

        if eef_wrench is not None:
            robot_joint_from_base_transformations = self.fk(q, kind='joints')
            last_joint_to_world = robot_joint_from_base_transformations[-1].inv
            last_joint_to_eef = Transformation(self.model.frames[self.tcp].placement).inv
            wrench_in_last_joint_frame = np.hstack((
                # force just needs rotation
                last_joint_to_world[:3, :3] @ eef_wrench[:3],
                # torque needs rotation + torques from offsetting force
                last_joint_to_world[:3, :3] @ eef_wrench[3:]
                + last_joint_to_world[:3, :3] @ (np.cross(last_joint_to_eef[:3, 3], eef_wrench[:3]))
            ))
            f_ext_vec.append(pin.Force(wrench_in_last_joint_frame))
        else:
            f = pin.Force()
            f.setZero()
            f_ext_vec.append(f)

        drift = pin.rnea(self.model, self.data, np.atleast_1d(q), np.atleast_1d(dq), np.atleast_1d(ddq), f_ext_vec)
        return drift + motor_inertia * self.motor_inertias(ddq) + friction * self.friction(dq)

    def ik(self,
           eef_pose: ToleratedPose,
           q_init: np.ndarray = None,
           **kwargs) -> Tuple[np.ndarray, bool]:
        """
        Default ik to jacobian ik. See ik_jacobian for more documentation

        :param eef_pose: The desired 4x4 placement of the end effector
        :param q_init: The joint configuration to start with. If not given, will start the iterative optimization at the
            current configuration.
        :return: A tuple of (q_solution, success [boolean])
        """
        if kwargs.get('ignore_self_collision', False) and 'task' in kwargs:
            raise ValueError("Checking for collisions in a task without checking for self-collisions is not supported.")

        if self.njoints < 6:
            # Jacobian unstable for small robots
            return self.ik_scipy(eef_pose, q_init, **kwargs)
        else:
            return self.ik_jacobian(eef_pose, q_init, **kwargs)

    def ik_jacobian(self,
                    eef_pose: ToleratedPose,
                    q_init: np.ndarray = None,
                    gain: Union[float, np.ndarray] = 1.,
                    damp: float = 1e-12,
                    max_iter: int = 1000,
                    kind='damped_ls_inverse',
                    **kwargs) -> Tuple[np.ndarray, bool]:
        """
        Implements numerics inverse kinematics solvers based on the jacobian.

        If no solution is found, returns success=False and the configuration q with the lowest euclidean distance to the
        desired pose (ignoring orientation offset) that was found during max_iter iterations.

        Reference: Robotics, Siciliano et al. [2009], Chapter 3.7

        :param eef_pose: The desired 4x4 placement of the end effector
        :param q_init: The joint configuration to start with. If not given, will start the iterative optimization at the
            current configuration.
        :param gain: Gain Matrix K for closed "control" of q. Higher K leads to faster, but instable solutions. This is
            only used for "transpose" or "pseudo_inverse" methods.
        :param damp: Damping for the damped least squares pseudoinverse method
        :param max_iter: The maximum number of iterations before the algorithm fails.
            (The default value 1000 is very conservative and can often be reduced by orders of magnitude;
            run with logging set to debug to see how often ik fails due to hitting max iter for your problem)
        :param kind: One of "transpose", "pseudo_inverse", "damped_ls_inverse". The according variant of the Jacobian
            will be used to solve the ik problem. While the transpose is the fastest one to calculate, for complicated
            problems it is too inaccurate to converge (quickly) to a solution. Pseudo inverse and damped least squares
            pseudo-inverse are pretty similar, the latter one introduces a "punishment" for the norm of the joint
            velocities and is therefore more resistant against singularities.
        :param kwargs: Additional arguments to be passed to the jacobian function. These include:
            - task: If a timor.Task.Task is provided here, the ik will restart if there are task collisions as long as
                max_iter is not reached.
            - joint_mask: A boolean array of length njoints that indicates which joints should be considered for the IK
            - allow_random_restart: The IK will restart from random configurations if it fails to converge. This can be
                disabled by setting this to False.
            - ignore_self_collision: If True, the IK will ignore self-collisions
            - check_static_torques: If True, the IK will check if the torques are within the static torque limits. If
                not so, the IK will restart from a random configuration as long as max_iter allows.
        :return: A tuple of q_solution [np.ndarray], success [boolean]
        """
        # Custom errors to make sure no more deprecated arguments are used for the IK solver
        if 'tolerance' in kwargs:
            raise ValueError("The placement needs to be tolerated - providing an extra tolerance is deprecated.")

        if self.njoints < 1:
            return np.array([]), eef_pose.valid(self.fk())

        rot_joint_mask = np.array([jnt.shortname() == 'JointModelRZ' for jnt in self.model.joints[1:]], dtype=bool)
        if 'joint_mask' in kwargs:
            rot_joint_mask = np.logical_and(rot_joint_mask, kwargs['joint_mask'])
        if q_init is None:
            if 'joint_mask' in kwargs:
                raise ValueError("joint_mask is only supported if you provide a q_init")
            # Find an initial guess for the solution where the robot is close to the desired point in space
            candidates = [self.configuration] + [self.random_configuration() for _ in range(4)]
            distances = [self.fk(cand).distance(eef_pose.nominal).translation_euclidean for cand in candidates]
            q = candidates[np.argmin(distances)]
        else:
            q = q_init
        joint_idx_pin = self.model.frames[self.tcp].parent
        desired = pin.SE3(eef_pose.nominal.homogeneous)
        joint_desired = desired.act(self.model.frames[self.tcp].placement.inverse())

        def inv(J: np.ndarray):
            if kind == 'transpose':
                return J.T
            elif kind == 'pseudo_inverse':
                return np.linalg.pinv(J)
            else:
                raise ValueError("Unknown argument, kind={}".format(kind))

        i = 1
        success = True
        closest_translation = kwargs.get('closest_translation_q', IntermediateIkResult(q, -np.inf))

        def random_restart(iter_left: int) -> Tuple[np.ndarray, bool]:
            """
            Evaluate a random restart given the number of iterations left and whether restarts are allowed.

            :param iter_left: number of iterations left till iter_max
            :return: Same as parent method
            """
            if kwargs.get("allow_random_restart", False):
                return closest_translation.q, False
            new_init = self.random_configuration()
            if 'joint_mask' in kwargs:
                inverted_mask = ~kwargs['joint_mask'].astype(bool)
                new_init[inverted_mask] = q_init[inverted_mask]
            return self.ik_jacobian(eef_pose, new_init, gain, damp, iter_left, kind, **kwargs)

        while not eef_pose.valid(self.fk(q)):
            joint_current = self.data.oMi[joint_idx_pin]
            diff = joint_current.actInv(joint_desired)
            error_twist = pin.log(diff).vector
            abs_translational_distance = np.linalg.norm(diff.translation)
            if abs_translational_distance < closest_translation.distance \
                    and self.q_in_joint_limits(q) \
                    and not self.has_self_collision(q):
                closest_translation = IntermediateIkResult(q, abs_translational_distance)
            J = pin.computeJointJacobian(self.model, self.data, q, joint_idx_pin)
            try:
                if kind == 'damped_ls_inverse':
                    q_dot = np.linalg.lstsq(J, error_twist, rcond=damp)[0]  # Damped least squares
                else:
                    J_inv = inv(J)  # Analytical Jacobian "pseudo inverse" (or transpose)
                    q_dot = J_inv.dot(gain).dot(error_twist)
            except np.linalg.LinAlgError:
                logging.debug(f"Jacobian ik break due to singularity after {i} iter for q={q}. Trying again.")
                kwargs['closest_translation_q'] = closest_translation
                return random_restart(max_iter - i)
            if 'joint_mask' in kwargs:
                q_dot = q_dot * kwargs['joint_mask']
            q = pin.integrate(self.model, q, q_dot)
            # Python modolo defaults to positive values, but we want to preserve q sign
            sign_preserve = np.ones_like(q) * 2 * np.pi
            sign_preserve[q >= 0] = 0
            # Keep joint angles (for revolute joints) in interval (-2pi, 2pi)
            q[rot_joint_mask] = q[rot_joint_mask] % (2 * np.pi) - sign_preserve[rot_joint_mask]

            if np.inf in q:
                logging.debug(f"Jacobian ik break due to q approaching inf after {i} iter. Trying again.")
                kwargs['closest_translation_q'] = closest_translation
                return random_restart(max_iter - i)
            if i >= max_iter:
                logging.debug(f"Jacobian ik not successful due to reaching the maximum number of iterations {i}")
                logging.debug("Returning partial/closest solution.")
                return closest_translation.q, False
            i += 1

        if not self.q_in_joint_limits(q):
            if not kwargs.get('allow_wrapping_joint_angle', True):
                logging.debug("Wrapping forbidden; retry solving ik.")
                return random_restart(max_iter - i)

            q[rot_joint_mask] = spatial.rotation_in_bounds(q[rot_joint_mask], self.joint_limits[:, rot_joint_mask])
            if self.q_in_joint_limits(q):
                # We're good to go
                logging.debug("Successfully mapped ik result to the robot joint limits")
            elif i < max_iter:
                # Give it another try, starting from a different configuration
                i += 1
                logging.info("IK was out of joint limits, re-try")
                kwargs['closest_translation_q'] = closest_translation
                return random_restart(max_iter - i)
            else:
                logging.debug("IK was out of joint limits, but we're out of iterations")

        if kwargs.get('check_static_torques', False):
            tau = self.static_torque(q)
            if not (np.abs(tau) <= self.joint_torque_limits).all():
                logging.debug("IK was out of joint torque limits, re-try")
                kwargs['closest_translation_q'] = closest_translation
                return random_restart(max_iter - i)

        if self.has_self_collision(q) and not kwargs.get('ignore_self_collision', False):
            logging.info("IK solution had self-collisions, re-try")
            if i < max_iter:
                # Give it another try, starting from a different configuration
                i += 1
                kwargs['closest_translation_q'] = closest_translation
                return random_restart(max_iter - i)
            logging.debug("Jacobian ik not successful after maximum number of iterations, fails with self collision.")
            success = False
        elif 'task' in kwargs and self.has_collisions(kwargs['task']):
            logging.info("IK solution collides with environment, re-try")
            if i < max_iter:
                # Give it another try, starting from a different configuration
                i += 1
                logging.debug("Keep the previously best q, but without checking for environment collisions.")
                kwargs['closest_translation_q'] = closest_translation
                return random_restart(max_iter - i)
            logging.debug("Jacobian ik not successful after maximum number of iterations, fails due to collision.")
            success = False

        logging.debug(f"Jacobian ik successful after {i} iterations")
        return q, success

    def move(self, displacement: TransformationLike):
        """
        Moves frames, joints and geometries connected to the world.

        :param displacement: Transformation / transformation transform to move, given in the world/universe frame
        """
        displacement = pin.SE3(Transformation(displacement, set_safe=True).homogeneous)
        for i, frame in enumerate(self.model.frames):
            if frame.type == pin.FrameType.JOINT:
                break
            cos = spatial.frame2geom(i, self.collision)
            vos = spatial.frame2geom(i, self.visual)
            for co in cos:
                co.placement = displacement * co.placement
            for vo in vos:
                vo.placement = displacement * vo.placement
            frame.placement = displacement * frame.placement
        self.model.jointPlacements[0] = displacement * self.model.jointPlacements[0]
        for i in range(1, self.njoints + 1):
            # Move all joints that are directly connected to "world" (i.e. the first joint[s])
            if self.model.parents[i] != 0:
                continue
            try:
                self.model.jointPlacements[i] = displacement * self.model.jointPlacements[i]
            except IndexError:
                assert self.njoints == 0, "Could not find the first joint, but there seems to be one"
        self.update_configuration(self.configuration)
        self._base_placement = Transformation(self.placement.multiply_from_left(displacement.homogeneous),
                                              set_safe=True)

    def random_configuration(self) -> np.ndarray:
        """Generates a random configuration."""
        try:
            return pin.randomConfiguration(self.model)
        except RuntimeError:
            logging.debug('Random configuration failed due to infinite joint limits - setting to (-2pi, 2pi)')
            lower = np.maximum(self.joint_limits[0, :], -2 * np.pi)
            upper = np.minimum(self.joint_limits[1, :], 2 * np.pi)
            return (upper - lower) * np.random.rand(self.njoints) + lower

    def set_base_placement(self, placement: TransformationLike) -> None:
        """Moves the base to desired position"""
        placement = Transformation(placement)
        if placement != self._base_placement:
            self.move(placement @ self.placement.inv)

    def static_torque(self, q: np.ndarray, f_ext: pin.StdVec_Force = None) -> np.ndarray:
        """
        Static torques of the robot

        :param q: Current joint configuration
        :param f_ext: External forces as array of pin forces
        :return: n_joints x 1 array of torques
        """
        if f_ext is None:
            return pin.computeGeneralizedGravity(self.model, self.data, q)
        else:  # To be tested
            logging.warning("Static torques with external forces are not properly tested yet")
            return pin.computeStaticTorque(self.model, self.data, q, f_ext)

    def update_configuration(self,
                             q: np.ndarray,
                             dq: np.ndarray = None,
                             ddq: np.ndarray = None,
                             *,
                             frames: bool = False,
                             geometry: bool = False):
        """
        Update the robot configuration by updating the joint parameters.

        :param q: The new configuration in joint space iven as numpy array of joint values.
        :param dq: Joint velocities, optional
        :param ddq: Joint acceleration, optional
        :param frames: If true, frame placements are also updated
        :param geometry: If true, geometry (visual and collision) placements are updated
        :return: None
        """
        if dq is None and ddq is not None:
            raise ValueError("If ddq is specified, dq must be specified as well")
        self.configuration = q
        if dq is not None:
            self.velocities = dq
            if ddq is None:
                pin.forwardKinematics(self.model, self.data, self.configuration, dq)
                self.accelerations = np.zeros_like(self.q)
            else:
                self.accelerations = ddq
                pin.forwardKinematics(self.model, self.data, self.configuration, dq, ddq)
        else:
            self.accelerations = np.zeros_like(self.q)
            self.velocities = np.zeros_like(self.q)
            pin.forwardKinematics(self.model, self.data, self.configuration)
        if frames:
            pin.updateFramePlacements(self.model, self.data)
        if geometry:
            self._update_geometry_placement()

    # ---------- Collision Methods ----------

    def __check_collisions(self,
                           task: 'Task.Task',  # noqa: F821
                           return_at_first: bool
                           ) -> Union[bool, List[Tuple[RobotBase, Union[RobotBase, Obstacle]]]]:
        """
        Checks all collision pairs or all collisions pairs until the first colliding one is found.

        :param task: The task to check for collisions
        :param return_at_first: If true, this function behaves like a "has_collision". If false, it behaves like a
            "get_all_collisions" function.
        :return: Bool if return_at_first is true, otherwise a list of all colliding pairs.
        """
        collisions: List[Tuple[RobotBase, Union[RobotBase, Obstacle]]] = list()
        request = CollisionRequest()
        result = CollisionResult()

        if self.has_self_collision():
            if return_at_first:
                return True
            collisions.append((self, self))

        geometries = [rco.geometry for rco in self.collision_objects]
        for obstacle in task.obstacles:
            for (robot_geometry, omg), (t, geometry) in \
                    itertools.product(zip(geometries, self.collision_data.oMg), obstacle.collision.collision_data):
                omg3f = Transform3f(omg.homogeneous[:3, :3], omg.homogeneous[:3, 3])
                t3f = Transform3f(t[:3, :3], t[:3, 3])
                if collide(robot_geometry, omg3f, geometry, t3f, request, result):
                    if return_at_first:
                        return True
                    collisions.append((self, obstacle))
                    break  # Next obstacle
        if return_at_first:
            return False
        return collisions

    def collisions(self,
                   task: 'Task.Task'  # noqa: F821
                   ) -> List[Tuple[RobotBase, Union[RobotBase, Obstacle]]]:
        """
        Returns all collisions that appear between the robot and itself or obstacles in the task.

        :param task: The task to check for collisions
        :return: A list of tuples (robot, obstacle) for robot an obstacle in collision
        """
        return self.__check_collisions(task, return_at_first=False)

    def has_collisions(self, task: 'Task.Task') -> bool:  # noqa: F821
        """
        Returns true if there is at least one collision in the task or one self-collision.

        :param task: The task the robot is in.
        :return: Boolean indicator whether there are any collisions.
        """
        return self.__check_collisions(task, return_at_first=True)

    def has_self_collision(self, q: np.ndarray = None) -> bool:
        """
        Returns true if there are any self collisions in the current robot configuration.

        If q is provided, the robot configuration will be changed. If not, this defaults to the current configuration.
        """
        if q is not None:
            self.update_configuration(q)
        self._update_collision_placement()
        return pin.computeCollisions(self.model, self.data, self.collision, self.collision_data, self.configuration,
                                     True)

    def _init_collision_pairs(self):
        """
        Adds all collision pairs to the robot.
        """
        self.collision.addAllCollisionPairs()

    def _remove_home_collisions(self):
        """
        Removes all collision pairs that are in collision while the robot is in its current (usually home) configuration

        This can be replaced by manually defining collision pairs to be checked or reading an SRDF file that defines the
        collision pairs that can be ignored.
        Keep in mind that this method leads to never detecting collisions that already exist in the current
        configuration of the robot.
        """
        # TODO: Implement more precise methods, e.g. SRDF input
        self._update_collision_placement()
        pin.computeCollisions(self.model, self.data, self.collision, self.collision_data, self.configuration, False)
        active = np.zeros((self.collision.ngeoms, self.collision.ngeoms), bool)
        for k in range(len(self.collision.collisionPairs)):
            cr = self.collision_data.collisionResults[k]
            cp = self.collision.collisionPairs[k]
            if cr.isCollision():
                active[cp.first, cp.second] = False
            else:
                active[cp.first, cp.second] = True

        self.collision.setCollisionPairs(active)
        self.collision_data = self.collision.createData()
        assert not self.has_self_collision(), "Self collisions should have been removed by now..."

    def _update_collision_placement(self):
        """Updates the placement of collision geometry objects"""
        pin.updateGeometryPlacements(self.model, self.data, self.collision, self.collision_data)

    # ---------- Visualization ----------
    def _update_visual_placement(self):
        """Updates the placement of visual geometry objects"""
        pin.updateGeometryPlacements(self.model, self.data, self.visual, self.visual_data)

    def visualize(self,
                  visualizer: pin.visualize.BaseVisualizer = None,
                  coordinate_systems: Union[str, None] = None,
                  update_collision_visuals: Optional[bool] = True,
                  ) -> pin.visualize.MeshcatVisualizer:
        """
        Displays the robot in its current environment using a Meshcat Visualizer in the browser

        :param visualizer: A meshcat visualizer instance. If none, a new will be created
        :param coordinate_systems: Can be None, 'tcp', 'joints' or 'full' - the specified coordinates will be drawn
            (full means all frames, as in fk).
        :param update_collision_visuals: If true, the collision geometry will be updated as well
          (but not shown by default).
        :return: The meshcat visualizer instance used for plotting the robot
        """
        if coordinate_systems not in (None, 'joints', 'full', 'tcp'):
            raise ValueError("Invalid Value for coordinate system argument: {}".format(coordinate_systems))

        if visualizer is None:
            visualizer = pin.visualize.MeshcatVisualizer(
                self.model, self.collision, self.visual,
                copy_models=False,
                data=self.data,
                collision_data=self.collision_data if update_collision_visuals else None,
                visual_data=self.visual_data)
        elif visualizer.model is not self.model:
            # Copy everything, s.t. robot updates are transferred to the visualizer!
            visualizer.model = self.model
            visualizer.collision_model = self.collision
            visualizer.visual_model = self.visual
            visualizer.data = self.data
            visualizer.collision_data = self.collision_data if update_collision_visuals else None
            visualizer.visual_data = self.visual_data

        if not hasattr(visualizer, 'viewer'):
            visualizer.initViewer()
        visualizer.loadViewerModel(rootNodeName=self.name)

        # Force update of collision meshes; otherwise toggle shows them not at current configuration
        if update_collision_visuals:
            visualizer.updatePlacements(pin.COLLISION)

        visualizer.display(self.configuration)
        if coordinate_systems is not None:
            if coordinate_systems == 'tcp':
                self.fk(kind='tcp').visualize(visualizer, name='tcp', scale=.3)
            else:
                names = self.joints if coordinate_systems == 'joints' else (fr.name for fr in self.model.frames)
                for transform, name in zip(self.fk(kind=coordinate_systems), names):
                    transform.visualize(visualizer, name=name, scale=.3)

        if not update_collision_visuals:
            logging.debug("Collisions toggle in meshcat interface will show collision meshes only at origin.")

        return visualizer

    def visualize_self_collisions(self, visualizer: pin.visualize.MeshcatVisualizer = None) \
            -> pin.visualize.MeshcatVisualizer:
        """
        Computes all collisions and visualizes them in a different color for every collision pair in contact.

        :note: **Only** considers the pre-defined collision pairs.

        :param visualizer: A pinnocchio meshcat visualizer (only tested for this one)
        """
        # Compute collisions already updates geometry placement
        pin.computeCollisions(self.model, self.data, self.collision, self.collision_data, self.configuration, False)

        collides = list()
        for k in range(len(self.collision.collisionPairs)):
            cr = self.collision_data.collisionResults[k]
            first, second = map(int, self.collision_pairs[k])
            if cr.isCollision():
                collides.append([self.collision.geometryObjects[first], self.collision.geometryObjects[second]])
                print("collision pair detected:", first, ",", second, "- collision:",
                      "Yes" if cr.isCollision() else "No")

        if visualizer is None:
            visualizer = pin.visualize.MeshcatVisualizer(self.model, self.collision, self.visual,
                                                         copy_models=False,
                                                         data=self.data,
                                                         collision_data=self.collision_data,
                                                         visual_data=self.visual_data)
        elif visualizer.model != self.model:
            visualizer.model = self.model
            visualizer.collision_model = self.collision
            visualizer.visual_model = self.visual
            visualizer.rebuildData()

        if not hasattr(visualizer, 'viewer'):
            visualizer.initViewer()
        visualizer.loadViewerModel(rootNodeName=self.name)

        # red = np.array([1, 0.2, 0.15, 1])
        collision_colors = [np.hstack([np.array(clr), np.ones(1)]) for clr in colors.BASE_COLORS.values()]
        visualizer.displayCollisions(True)
        visualizer.displayVisuals(False)
        visualizer.display(self.configuration)
        for i, cp in enumerate(collides):
            clr = collision_colors[i % len(collision_colors)]
            cp[0].meshColor = clr
            cp[1].meshColor = clr
            visualizer.loadViewerGeometryObject(cp[0], pin.GeometryType.COLLISION)
            visualizer.loadViewerGeometryObject(cp[1], pin.GeometryType.COLLISION)

        return visualizer

    # ---------- Morphology / Configuration changes ----------
    def _add_body(self,
                  body: PinBody,
                  parent_joint: Union[int, str],
                  parent_frame_id: int
                  ) -> Tuple[int, Tuple[int, ...]]:
        """
        Adds a single body to the robot. Note that collision and visual data are not updated!

        :param body: A body wrapper
        :param parent_joint: The index or name of the parent joint
        :param parent_frame_id: The id of the parent frame of the body (important to calculate body placement)
        :return: The index of the new body frame and indices for the newly added collision geometries
        """
        if isinstance(parent_joint, str):
            parent_joint = self.model.getJointId(parent_joint)
        # Then the bodies belonging to it
        self.model.appendBodyToJoint(**body.body_data(parent_joint))
        bf_kwargs = body.body_frame_data(parent_joint, parent_frame_id)
        new_frame_id = self.model.addBodyFrame(
            bf_kwargs['body_name'],
            bf_kwargs['parentJoint'],
            bf_kwargs['body_placement'],
            bf_kwargs['previous_frame']
        )

        # And also the geometry objects belonging to the bodies
        geometry_indices = list()
        for cg in body.collision_geometries:
            cg.parentJoint = bf_kwargs['parentJoint']
            cg.parentFrame = new_frame_id
            cg.placement = bf_kwargs['body_placement'].act(cg.placement)
            geometry_indices.append(self.collision.addGeometryObject(cg))
        for vg in body.visual_geometries:
            vg.parentJoint = bf_kwargs['parentJoint']
            vg.parentFrame = new_frame_id
            vg.placement = bf_kwargs['body_placement'].act(vg.placement)
            self.visual.addGeometryObject(vg)

        return new_frame_id, tuple(geometry_indices)

    def _add_joint(self,
                   joint: pin.JointModel,
                   parent: Union[int, str],
                   bodies: Union[PinBody, Collection[PinBody]] = (),
                   fixed_joints: Union[Tuple[pin.Frame, ...], List[pin.Frame]] = (),
                   inertia: pin.Inertia = None,
                   update_kinematics: bool = True,
                   update_home_collisions: bool = True,
                   **kwargs) -> int:
        """
        Adds a joint to the current robot model.

        :param joint: A pinocchio joint model instance
        :param parent: Either the index or the name of the parent joint to which the new one shall be appended
        :param bodies: All bodies attached to the joint. To add N bodies, there need to be N-1 fixed joints.
        :param fixed_joints: Fixed joints attached to the joint. Assumes bodies and fixed joints are added in the order
            b1-fj1-b2-fx2-...-bn
        :param inertia: The inertia attached to the joint. If not provided, will be auto-calculated by the bodies added.
        :param update_kinematics: If False, the datas will not be re-calculated, meaning the changes to not have
            an immediate effect on the results of kinematic/dynamic methods. Is useful when adding multiple joints,
            so just after the last change the data needs to be updated.
        :param update_home_collisions: If False, the visual and collision geometries will not be updated,
            meaning this joint will neither be plotted nor considered for collision checking until done so.
            Useful for either building models that don't need either of these features or when adding multiple joints,
            so this can be done after the last only.
        :param kwargs: All key-word-arguments supported by pin.Model.addJoint:

            * joint_placement: [SE3] -> Placement of the joint inside its parent joint.
            * joint_name: [str] -> If empty, name will be random
            * max_effort: [float] -> Maximal Joint Torque
            * max_velocity: [float] -> Maximal Joint Velocity
            * min_config and max_config: [float] -> Joint configuration limits
            * friction: [float] -> Joint friction parameters
            * damping: [float] -> Joint damping parameters
        :return: Index of the new joint
        """
        if len(fixed_joints) > 0:
            if len(fixed_joints) != len(bodies) - 1:
                # Usually, fixed joints are only added in this method when copying another pin robot.
                # So far, I have never seen any pin models where the ordering was not body-fj-body-fj-...-body.
                # Theoretically, this does not need to be enforced, but then, this method must be adapted accordingly.
                raise ValueError("Number of fixed joints must be equal to number of bodies - 1, "
                                 "if fixed joints are provided")

        kwargs.setdefault('joint_placement', pin.SE3.Identity())
        kwargs.setdefault('joint_name', 'Joint_' + ''.join(np.random.choice(list(string.ascii_letters), 5)))

        for eigen_val in ['max_effort', 'max_velocity', 'min_config', 'max_config', 'friction', 'damping']:
            if eigen_val in kwargs:
                # The C++ bindings don't take floats here but 1D np arrays
                kwargs[eigen_val] = float2array(kwargs[eigen_val])

        if isinstance(bodies, PinBody):
            bodies = [bodies]

        if isinstance(parent, int):
            parent_id = parent
        elif isinstance(parent, str):
            parent_id = self.model.getJointId(parent)
        else:
            raise ValueError("Cannot interpret the parent parameter as it is neither the index nor the name.")

        gear_ratio = kwargs.pop('gearRatio', 1.)
        rotor_inertia = kwargs.pop('motorInertia', 0.)

        # First add the joint
        idx = self.model.addJoint(parent_id, joint, **kwargs)
        parent_frame_id = self.model.addJointFrame(idx)

        assert self.model.joints[idx].idx_q == self.model.joints[idx].idx_v, "nq != nv???"
        self.model.rotorGearRatio[self.model.joints[idx].idx_v] = gear_ratio
        self.model.rotorInertia[self.model.joints[idx].idx_v] = rotor_inertia

        for i, body in enumerate(bodies):
            if i > 0 and len(fixed_joints) > 0:  # Fixed Joints in between bodies
                fj_frame = pin.Frame(fixed_joints[i - 1])
                fj_frame.previousFrame = parent_frame_id
                fj_frame.parent = idx
                parent_frame_id = self.model.addFrame(fj_frame)
            parent_frame_id, _ = self._add_body(body, idx, parent_frame_id)

        if inertia is not None:
            # Overwrite the inertia defined by the bodies added
            self.model.inertias[idx] = inertia
        self.configuration = np.hstack([self.configuration, 0.])

        if update_kinematics and update_home_collisions:
            self._update_kinematic()
            self._remove_home_collisions()
        elif update_kinematics:
            self._update_kinematic()
        else:
            assert not update_home_collisions, "Cannot update geometries without updating kinematics"

        if self.model.frames[self.tcp].name != 'TCP':
            # No dedicated TCP frame yet, so defaulting to most recently added joint
            self.tcp = self.model.nframes - 1

        return idx

    def add_tcp_frame(self, parent_joint: str, placement: pin.SE3):
        """
        Adds a Tool Center Point that will be used for inverse kinematic calculations.

        :param parent_joint: The joint the TCP is attached to
        :param placement: Relative placement of the TCP in the parent joint frame
        :return: The index of the TCP frame
        """
        parent_frame = self.model.getFrameId(parent_joint)
        parent_joint_index = self.model.getJointId(parent_joint)
        tcp_frame = pin.Frame('TCP', parent_joint_index, parent_frame, placement, pin.FrameType.OP_FRAME)
        self.tcp = self.model.addFrame(tcp_frame)
        self.data = self.model.createData()
