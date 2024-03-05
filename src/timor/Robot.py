#!/usr/bin/env python3
# Author: Jonathan Külz
# Date: 12.01.22
from __future__ import annotations

import abc
import itertools
from pathlib import Path
import string
from typing import Callable, Collection, Dict, Iterable, List, Optional, Sequence, Tuple, Union
from warnings import warn

from hppfcl.hppfcl import CollisionObject, CollisionRequest, CollisionResult, Transform3f, collide
from matplotlib import colors
import numpy as np
import pinocchio as pin
from scipy.optimize import OptimizeResult
import scipy.optimize as optimize

from timor.Bodies import PinBody
from timor.task.Obstacle import Obstacle
from timor.utilities import logging, spatial
from timor.utilities.callbacks import CallbackReturn, IKCallback, chain_callbacks
from timor.utilities.configurations import IK_RANDOM_CANDIDATES
from timor.utilities.dtypes import IKMeta, IntermediateIkResult, float2array
from timor.utilities.frames import WORLD_FRAME
from timor.utilities.tolerated_pose import ToleratedPose
from timor.utilities.transformation import Transformation, TransformationLike


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

    _known_ik_solvers = {'scipy': 'ik_scipy'}

    def __init__(self,
                 *,
                 name: str,
                 home_configuration: np.ndarray = None,
                 base_placement: Optional[TransformationLike] = None,
                 rng: Optional[np.random.Generator] = None
                 ):
        """
        RobotBase Constructor

        :param home_configuration: Home configuration of the robot (default pin.neutral)
        :param base_placement: Home configuration of the robot (default pin.neutral)
        :param rng: Optionally, a numpy random number generator to use for anything randomized happening with this
            robot can be provided. This rng will be the default, but can be overridden by users explicitly desiring to
            use another one at times.
        """
        self._name: str = name
        if home_configuration is None:
            home_configuration = np.zeros((self.dof,))
        self.configuration: np.ndarray = np.empty_like(home_configuration)
        self.update_configuration(home_configuration)
        self.velocities: np.ndarray = np.zeros(self.configuration.shape)
        self.accelerations: np.ndarray = np.zeros(self.configuration.shape)

        if base_placement is not None:
            self._base_placement = Transformation(base_placement)
            self.move(base_placement)
        else:
            self._base_placement = Transformation.neutral()

        self._rng: np.random.Generator = np.random.default_rng() if rng is None else rng

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
    def dof(self) -> int:
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
        """Returns an array of shape 2xdof containing the lower and upper joint position limits"""

    @property
    @abc.abstractmethod
    def joint_acceleration_limits(self) -> np.ndarray:
        """Returns an array of shape 1xdof containing the absolute joint acceleration limits"""

    @property
    @abc.abstractmethod
    def joint_torque_limits(self) -> np.ndarray:
        """Returns an array of shape 1xdof containing the absolute joint torque limits"""

    @property
    @abc.abstractmethod
    def joint_velocity_limits(self) -> np.ndarray:
        """Returns an array of shape 1xdof containing the absolute joint velocity limits"""

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
    def collisions(self, task: 'Task.Task', safety_margin: float = 0) -> List[Tuple[RobotBase, Obstacle]]:  # noqa: F821
        """
        Returns all collisions that appear between the robot and itself or obstacles in the task.

        :param task: The task to check for collisions
        :param safety_margin: The safety margin is a padding added to all obstacles to increase needed separation.
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
    def has_collisions(self, task: 'Task.Task', safety_margin: float = 0) -> bool:  # noqa: F821
        """
        Returns true if there is at least one collision in the task or one self-collision.

        Faster than self.collisions in case of collision, as this method breaks as soon as the first one is detected.

        :param task: The task the robot is in.
        :param safety_margin: The safety margin is a padding added to all obstacles to increase needed separation.
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

        :param q: Dimension dofx1.
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
           *,
           allow_random_restart: bool = True,
           current_best: Optional[IntermediateIkResult] = None,
           ignore_self_collisions: bool = False,
           ik_cost_function: Callable[[RobotBase, np.ndarray, TransformationLike], float] = default_ik_cost_function,  # noqa: E501
           ik_method: str = 'scipy',
           inner_callbacks: Optional[Iterable[IKCallback]] = None,
           joint_mask: Optional[Sequence[bool]] = None,
           max_iter: int = 1000,
           outer_callbacks: Optional[Iterable[IKCallback]] = None,
           outer_constraints: Optional[Iterable['Constraints.RobotConstraint']] = None,  # noqa: F821
           q_init: Optional[np.ndarray] = None,
           task: Optional['Task.Task'] = None,  # noqa: F821
           convergence_threshold: float = 1e-8,
           **kwargs) -> Tuple[np.ndarray, bool]:
        """
        Try to find the joint angles q that minimize the error between TCP pose and the desired eef_pose.

        This serves as the general interface to ik methods. In general, these methods can be split in an inner and an
        outer loop. The inner loop (that could, theoretically also be a single step only) is responsible for (mostly
        numeric) optimization, while the outer loop is responsible for restarting the inner loop if it fails to
        converge or satisfy some constraints. This method is the outer loop, while the inner loop is determined by the
        ik_method argument.

        :param eef_pose: The desired 4x4 pose of the end effector.
        :param allow_random_restart: If True, the IK will restart from random configurations if it fails to converge.
            Disable this for smooth motions (so, if you are not interested in solutions far away from q_init).
        :param current_best: The current best solution, if any. In case of failure, the best intermediate solution will
            be returned
        :param ignore_self_collisions: If True, self collisions will be ignored during the IK. By default, a solution
            with self collisions will never be considered successful.
        :param inner_callbacks: Any number of callbacks that will be called after each iteration of the inner loop.
        :param ik_cost_function: A custom cost function that takes a robot, its joint angles, and a
            TransformationLike and returns a scalar cost indicating the quality of the configuration in solving the ik
            problem.
        :param ik_method: The method to use for the inner loop. Defaults to 'scipy'.
        :param joint_mask: A boolean array that indicates which joints should be considered for the IK. If an index
            evaluates to False, the joint will not be moved.
        :param max_iter: The maximum number of iterations before the algorithm gives up.
        :param outer_callbacks: Any number of callbacks that will be called after each iteration of the outer loop.
        :param outer_constraints: Any number of constraints that must be satisfied by the solution but can not be
            incorporated in the inner optimization. The constraints are validated using the task provided to the ik
            and a joint velocity of zero. Joint limits and self collisions will always be considered if not explicitly
            ignored. The constraints are internally converted to callbacks that return false if the constraint is hurt.
        :param task: If a task is provided, any of the constraints defined in the task will be considered outer
            constraints.
        :param convergence_threshold: A threshold for convergence: When the cost function stops improving by more than
            this threshold, the IK will terminate even without finding a solution.
        :param q_init: The joint configuration to start with. If not given, it is up to the downstream method to
            determine a suitable initial configuration.
        :param kwargs: Any keyword arguments that are specific to the inner loop method.
        :return: A tuple of (q_solution, success [boolean])
        """
        from timor.task.Constraints import CollisionFree, SelfCollisionFree, RobotConstraint

        if self.dof == 0:
            return self.configuration, eef_pose.valid(self.fk())

        if inner_callbacks is None:
            inner_callbacks = []
        if outer_callbacks is None:
            outer_callbacks = []
        if outer_constraints is None:
            outer_constraints = []
        else:
            outer_constraints = list(outer_constraints)

        if task is not None:
            outer_constraints.extend([c for c in task.constraints if isinstance(c, RobotConstraint)])

        if ignore_self_collisions:
            if any(isinstance(c, CollisionFree) for c in outer_constraints):
                raise ValueError("You can not ignore self collisions and require collision free at the same time.")
        elif not any(isinstance(c, (CollisionFree, SelfCollisionFree)) for c in outer_constraints):
            outer_constraints.append(SelfCollisionFree())

        # Argument checks and preprocessing
        if ik_method not in self._known_ik_solvers:
            raise ValueError(f"Unknown ik_method {ik_method}")
        _ik = getattr(self, self._known_ik_solvers[ik_method])

        if joint_mask is not None and q_init is None:
            raise ValueError("If you provide a joint mask, you need to provide a q_init as well.")

        if q_init is None:
            # Find an initial guess for the solution where the robot is close to the desired point in space
            candidates = [self.random_configuration() for _ in range(IK_RANDOM_CANDIDATES)]
            distances = [ik_cost_function(self, q, eef_pose.nominal.in_world_coordinates()) for q in candidates]
            q_init = candidates[np.argmin(distances)]

        info = IKMeta(dof=self.dof, max_iter=max_iter, intermediate_result=current_best, joint_mask=joint_mask,
                      task=task, allow_random_restart=allow_random_restart)

        q = self.configuration
        success = False

        def constraint_callback(c: RobotConstraint) -> IKCallback:
            """
            Converts a robot constraint to a callback. Constraints will never return a BREAK condition, just True/False
            """
            def callback(_q: np.ndarray):
                return CallbackReturn(c.check_single_state(task, self, _q, np.zeros_like(_q)))
            return callback

        outer_callback = chain_callbacks(*outer_callbacks, *(constraint_callback(c) for c in outer_constraints))

        while (max_iter > 0) and not success:
            q, success = _ik(eef_pose,
                             q_init=q_init,
                             info=info,
                             callbacks=inner_callbacks,
                             convergence_threshold=convergence_threshold,
                             ik_cost_function=ik_cost_function,
                             **kwargs)  # Execute the inner loop

            if success:
                if outer_callback(q) is CallbackReturn.TRUE:
                    logging.verbose_debug("IK successfully converged.")
                    break
                logging.verbose_debug("IK converged, but constraints are not satisfied. Restarting if possible.")
                success = False
            elif outer_callback(q) is CallbackReturn.BREAK:
                logging.verbose_debug("IK was interrupted by a callback.")
                break

            if not allow_random_restart:
                logging.verbose_debug("IK failed to converge and random restarts are disabled.")
                break

            q_new = self.random_configuration()
            if joint_mask is not None:
                q_init[info.joint_mask] = q_new[info.joint_mask]
            else:
                q_init = q_new

            if not success and info.max_iter >= max_iter:
                raise RuntimeError("Running the inner IK did not reduce the number of iterations left. Danger of"
                                   "infinite recursion")
            max_iter = info.max_iter
            if max_iter > 0:
                logging.verbose_debug(f"Restarting IK with new initial configuration and {max_iter} iterations left.")

        if not success:
            logging.info("IK failed to converge.")
            q = info.intermediate_result.q

        return q, success

    def ik_scipy(self,
                 eef_pose: ToleratedPose,
                 info: IKMeta,
                 q_init: Optional[np.ndarray] = None,
                 callbacks: Iterable[IKCallback] = (),
                 ik_cost_function: Callable[[RobotBase, np.ndarray, TransformationLike], float] = default_ik_cost_function,  # noqa: E501
                 convergence_threshold: float = 1e-8,
                 scipy_kwargs: Optional[Dict] = None
                 ) -> Tuple[np.ndarray, bool]:
        """
        Computes the inverse kinematics (joint angles for a given goal destination).

        This method runs a single numerical IK until it converges or fails due to constraints or the maximum number of
        iterations. As a user, usually you want to try multiple random restarts and use the generalized ik interface
        that still allows you to choose the scipy minimize method for the inner loop.

        :param eef_pose: Desired end effector pose with tolerances.
        :param info: An IKMeta object that contains relevant meta information about the IK problem and can be adapted
            during the inner IK process.
        :param q_init: Initial joint angles to start searching from. If None, uses a random configuration
        :param callbacks: An iterable of callbacks that are called after each iteration during optimization.
        :param ik_cost_function: A custom cost function that takes a robot, its joint angles, and a
            TransformationLike and returns a scalar cost indicating the quality of the configuration in solving the ik
            problem. If not given, the default cost function is used (equal weighting of .5 meter / 180 degree error).
        :param convergence_threshold: Threshold for termination. Optimization terminates when two consecutive iterates
            are improving the cost by less than this.
        :param scipy_kwargs: Any keyword arguments that should be passed to the scipy minimize method. If not given, the
            'L-BFGS-B' method with defaults will be used.
        :return: A tuple of q_solution [np.ndarray], success [boolean]. If success is False, q_solution is the
            configuration with minimal IK cost amongst all seen in between iterations of this method. The IK is
            successful if the tcp pose satisfies the tolerances of eef_pose and all callbacks return True.
        """
        if q_init is None:
            q_start = self.random_configuration(rng=self._rng)
        else:
            q_start = q_init

        q_masked = q_start[info.joint_mask]

        def unmask(q_partial):
            q_ik = q_start
            q_ik[info.joint_mask] = q_partial
            return q_ik

        def compute_cost(_q: np.ndarray):
            return ik_cost_function(self, unmask(_q), eef_pose.nominal.in_world_coordinates())

        callback = chain_callbacks(*callbacks)

        def scipy_callback(intermediate_result: OptimizeResult):
            """
            Transform the provided callback into a scipy callback

            :param intermediate_result: The name is enforced by scipy. It contains the current state of optimization.
            :raise StopIteration: If further progress is impossible
            """
            if not isinstance(intermediate_result, OptimizeResult):
                logging.warning("Using scipy<1.11 is deprecated for the Timor ik interface. If possible, update!")
                if not isinstance(intermediate_result, np.ndarray):
                    raise ValueError("Unknown callback return type for scipy IK: {}".format(type(intermediate_result)))
                q_res = unmask(intermediate_result)
                cost = ik_cost_function(self, q_res, eef_pose.nominal.in_world_coordinates())
            else:
                q_res = unmask(intermediate_result.x)
                cost = intermediate_result.fun

            ret = callback(q_res)
            # Only if our inner constraints hold and the callback returns true, we update the intermediate result
            if ret is CallbackReturn.TRUE and cost < info.intermediate_result.cost:
                info.intermediate_result.q = q_res
            if ret is CallbackReturn.BREAK:
                raise StopIteration("Callback requested to break the optimization")  # The scipy way of breaking

        bounds = optimize.Bounds(self.joint_limits[0, info.joint_mask], self.joint_limits[1, info.joint_mask])
        if scipy_kwargs is None:
            scipy_kwargs = {}
        if 'options' in scipy_kwargs:
            scipy_kwargs['options']['maxiter'] = info.max_iter
        else:
            scipy_kwargs = {'options': {'maxiter': info.max_iter}}
        scipy_kwargs.setdefault('method', 'L-BFGS-B')
        sol = optimize.minimize(
            compute_cost,
            q_masked,
            callback=scipy_callback,
            bounds=bounds,
            tol=convergence_threshold,
            **scipy_kwargs
        )
        if sol.nit == 0:
            logging.warning("The scipy IK solver converged after a single iteration.")
            info.max_iter = info.max_iter - 1  # Avoids infinite recursion
        info.max_iter = info.max_iter - sol.nit
        q_sol = unmask(sol.x)

        # sol.success gives information about successful termination, but we are not interested in that
        success = eef_pose.valid(self.fk(q_sol)) and callback(q_sol) is CallbackReturn.TRUE

        logging.verbose_debug("Scipy IK ends with message: %s", sol.message)
        return q_sol, success

    def random_configuration(self, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """
        Generates a random configuration.

        :param rng: Random number generator to use. If not given, this method uses the robot default rng.
        """
        if rng is None:
            rng = self._rng
        lower = np.maximum(self.joint_limits[0, :], -2 * np.pi)
        upper = np.minimum(self.joint_limits[1, :], 2 * np.pi)
        return (upper - lower) * rng.random(self.dof) + lower

    def tau_in_torque_limits(self, tau: np.ndarray) -> bool:
        """
        Returns true if the torque vector tau is within the torque limits of the robot.

        We assume that the torque limits of the robot are symmetric (tau_min = -tau_max)

        :param tau: A torque vector of dimension dof x 1. The limits will be checked against abs(tau)
        """
        return bool((np.abs(tau) <= self.joint_torque_limits).all())  # bool() to convert np.bool to python bool

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

    _known_ik_solvers = {'scipy': 'ik_scipy', 'jacobian': 'ik_jacobian'}

    def __init__(self,
                 *,
                 wrapper: pin.RobotWrapper,
                 home_configuration: np.ndarray = None,
                 base_placement: TransformationLike = None,
                 rng: Optional[np.random.Generator] = None
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
                         base_placement=None,
                         rng=rng)

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
    def dof(self) -> int:
        """Number of actuated degrees of freedom."""
        return self.model.nq

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
        return np.ones((self.dof,), float) * np.inf

    @property
    def joint_torque_limits(self) -> np.ndarray:
        """Numpy array where arr[0,:] are the lower joint limits and arr[1,:] are the upper torque limits."""
        return self.model.effortLimit

    @property
    def joint_velocity_limits(self) -> np.ndarray:
        """Numpy array with upper bound absolute joint velocity limits."""
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

    def _resize_for_pinocchio(self, arr: Optional[np.ndarray]):
        """
        A helper function to make sure anything that can be expressed as (nDOF,) array is expressed as such.

        Pinocchio can't handle (nDOF, 1) arrays, so we need to squeeze them.
        :param arr: A numpy array. If this is None, this method will automatically return an array of zeros.
        """
        if arr is None:
            return np.zeros((self.dof,))
        return np.reshape(arr, (self.dof,))

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

        J = pin.computeFrameJacobian(self.model, self.data, q, self.tcp)
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

        :param motor_inertia: Whether to consider torques to balance motor side inertia. Even if set to true, the
            coriolis forces induced by the fast-rotating motors are not considered.
        :param friction: Whether to consider friction compensating torques
        :param eef_wrench: [f_x, f_y, f_z, tau_x, tau_y, tau_z] acting on the end-effector
        :param eef_wrench_frame: Frame the eef_wrench is described in (still acting on eef position!);
            for now only "world" accepted. TODO : Extend with future frame definitions
        """
        q = self.configuration if q is None else q
        dq = self.velocities if dq is None else dq
        ddq = self.accelerations if ddq is None else ddq
        q, dq, ddq = map(self._resize_for_pinocchio, (q, dq, ddq))
        known_frames = {"world"}  # TODO : Extend with future frame definitions
        if eef_wrench_frame not in known_frames:
            raise ValueError(f"{eef_wrench_frame} not in known frames {known_frames}")
        # pin expects (nDoF,) array but 1D robot can produce scalar q's
        f_ext_vec = pin.StdVec_Force()  # Force acting in each _joint_ frame
        for i in range(self.dof):
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

    def ik(self, eef_pose: ToleratedPose, *, ik_method: Optional[str] = None, **kwargs) -> Tuple[np.ndarray, bool]:
        """
        Interface for inverse kinematics solver.

        :param eef_pose: The desired 4x4 placement of the end effector
        :param ik_method: The method to use for the inner loop. Defaults to jacobian for robots with >=6 dof and scipy
            for robots with <6 dof.
        :return: A tuple of (q_solution, success [boolean])
        """
        if ik_method is None:
            if self.dof < 6:
                ik_method = 'scipy'
            else:
                ik_method = 'jacobian'
        return super().ik(eef_pose, ik_method=ik_method, **kwargs)

    def ik_jacobian(self,
                    eef_pose: ToleratedPose,
                    info: IKMeta,
                    q_init: np.ndarray,
                    callbacks: Iterable[IKCallback] = (),
                    damp: float = 1e-12,
                    ik_cost_function: Callable[[RobotBase, np.ndarray, TransformationLike], float] = default_ik_cost_function,  # noqa: E501
                    convergence_threshold: float = 1e-8,
                    alpha_average: float = .5,
                    error_term_mask: Optional[np.ndarray] = None,
                    gain: Union[float, np.ndarray] = .5,
                    kind='damped_ls_inverse',
                    tcp_or_world_frame: str = 'tcp',
                    ) -> Tuple[np.ndarray, bool]:
        """
        Implements numerics inverse kinematics solvers based on the jacobian.

        This method runs a single numerical IK until it converges or fails due to constraints or the maximum number of
        iterations. As a user, usually you want to try multiple random restarts and use the generalized ik interface
        that still allows you to choose the jacobian method for the inner loop.

        If no solution is found, returns success=False and the configuration q with the lowest euclidean distance to the
        desired pose (ignoring orientation offset) that was found during max_iter iterations.

        Reference: Robotics, Siciliano et al. [2009], Chapter 3.7
        Reference: https://gepettoweb.laas.fr/doc/stack-of-tasks/pinocchio/master/doxygen-html/md_doc_b-examples_i-inverse-kinematics.html  # noqa: E501
        Reference: https://scaron.info/robotics/inverse-kinematics.html

        :param eef_pose: Desired end effector pose with tolerances.
        :param info: An IKMeta object that contains relevant meta information about the IK problem and can be adapted
            during the inner IK process.
        :param q_init: The joint configuration to start with.
        :param callbacks: An iterable of callbacks that are called after each iteration during optimization.
        :param damp: Damping for the damped least squares pseudoinverse method
        :param ik_cost_function: While this cost function is not directly minimized, it is used to determine the
            quality of intermediate solutions and initial guesses if q_init is not provided.
        :param convergence_threshold: Threshold for termination. Optimization terminates when the exponentially decaying
            average of the cost function improvements is below this threshold.
        :param alpha_average: This factor controls the exponential decay of the running average computation, needed
            for the "convergence_threshold" stopping criterion. It must be between 0 and 1, where alpha=1 means no
            averaging at all and alpha close to 0 means averaging over many iterations. (0=average over all iterations)
        :param error_term_mask: A (6,) boolean mask that determines which components of the error term are considered.
            Depending on tcp_or_world_frame, the error term is either a [v, w] twist in the tcp frame or a
            [x, y, z, alpha, beta, gamma] vector in the world, where alpha, beta, gamma are the axis angles of the
            rotation. For example, masking the last three error terms [True, True, True, False, False, False] results in
            a partial inverse kinematics problem that only considers the position of the tcp.
        :param gain: Gain Matrix K determining the step size :math:`K J^{-q} e`.
        :param kind: One of "transpose", "pseudo_inverse", "damped_ls_inverse". The according variant of the Jacobian
            will be used to solve the ik problem. While the transpose is the fastest one to calculate, for complicated
            problems it is too inaccurate to converge (quickly) to a solution. Pseudo inverse and damped least squares
            pseudo-inverse are pretty similar, the latter one introduces a "punishment" for the norm of the joint
            velocities and is therefore more resistant against singularities.
        :param tcp_or_world_frame: Determines in which frame the Jacobian and the error term are computed. While tcp
            frame should be preferred for most applications, world frame allows for solving partial inverse kinematics
            for under-determined problems. See "error_term_mask" for more details.
        :return: A tuple of q_solution [np.ndarray], success [boolean]. If success is False, q_solution is the
            configuration with minimal IK cost amongst all seen in between iterations of this method. The IK is
            successful if the tcp pose satisfies the tolerances of eef_pose and all callbacks return True.
        """
        # For revolute joints, we enforce the joint angles to be in the interval (-2pi, 2pi)
        rot_joint_mask = np.array([jnt.shortname() == 'JointModelRZ' for jnt in self.model.joints[1:]], dtype=bool)
        rot_joint_mask = np.logical_and(rot_joint_mask, info.joint_mask)

        callback = chain_callbacks(*callbacks)

        if error_term_mask is None:
            error_term_mask = np.ones((6,), dtype=bool)
        if not 0 < alpha_average <= 1 and convergence_threshold < np.inf:
            raise ValueError("alpha_average must be in (0, 1]")

        if tcp_or_world_frame not in ['tcp', 'world']:
            raise ValueError("tcp_or_world_frame must be either 'tcp' or 'world'")

        # We solve the IK for the last joint instead of the TCP to save computation time
        desired = pin.SE3(eef_pose.nominal.in_world_coordinates().homogeneous)

        def inv(J: np.ndarray):
            if kind == 'transpose':
                return J.T
            elif kind == 'pseudo_inverse':
                return np.linalg.pinv(J)
            else:
                raise ValueError("Unknown argument, kind={}".format(kind))

        joint_mask = info.joint_mask.copy()
        success = False
        mean_improvement = 0
        previous_cost = ik_cost_function(self, q_init, eef_pose.nominal.in_world_coordinates())
        q = q_init
        self.update_configuration(q, frames=True, geometry=True)
        i = 0
        for i in range(info.max_iter):
            current = self.data.oMf[self.tcp]
            if tcp_or_world_frame == 'tcp':
                diff = current.actInv(desired)
                error = pin.log(diff).vector  # twist
                J = pin.computeFrameJacobian(self.model, self.data, q, self.tcp, pin.ReferenceFrame.LOCAL)
                J = pin.Jlog6(diff.inverse()) @ J  # Transform the jacobian to the joint frame
            else:
                current = Transformation(current)
                diff_world = WORLD_FRAME.rotate_to_this_frame(current.inv @ Transformation(desired), current)
                error = np.concatenate([diff_world.translation, diff_world.projection.axis_angles3])
                J = pin.computeFrameJacobian(self.model, self.data, q, self.tcp,
                                             pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)

            J = J[:, joint_mask]  # Only consider the joints we want to or can move
            try:
                if kind == 'damped_ls_inverse':
                    _q_dot = np.linalg.lstsq(J[error_term_mask], np.dot(gain, error[error_term_mask]), rcond=damp)[0]
                else:
                    J_inv = inv(J[error_term_mask])  # Analytical Jacobian "pseudo inverse" (or transpose)
                    _q_dot = J_inv.dot(gain).dot(error[error_term_mask])
            except (np.linalg.LinAlgError, SystemError):
                logging.verbose_debug(f"Jacobian ik break due to singularity after {i} iter for q={q}. Trying again.")
                break
            q_dot = np.zeros((self.dof,), float)
            q_dot[joint_mask] = _q_dot
            # For large angles, the linear approximation of revolute movements is not valid anymore
            q_dot[rot_joint_mask] = np.clip(q_dot[rot_joint_mask], -np.pi / 4, np.pi / 4)
            q = pin.integrate(self.model, q, q_dot)

            # Keep joint angles (for revolute joints) in interval (-2pi, 2pi) while preserving q sign
            sign_preserve = np.ones_like(q) * 2 * np.pi
            sign_preserve[q >= 0] = 0
            q[rot_joint_mask] = q[rot_joint_mask] % (2 * np.pi) - sign_preserve[rot_joint_mask]

            if not self.q_in_joint_limits(q):
                # We hinder the joint from moving any further and try to optimize for the others only
                keep_moving = np.logical_and(self.joint_limits[0, :] <= q, q <= self.joint_limits[1, :])
                joint_mask = np.logical_and(joint_mask, keep_moving)
                q = np.clip(q, self.joint_limits[0, :], self.joint_limits[1, :])

            if np.any(np.logical_or(np.isnan(q), np.isinf(q))):
                logging.verbose_debug(f"Jacobian ik break due to bad q: {q} after {i} iter. Trying again.")
                break

            cb_ret = callback(q)
            if cb_ret is CallbackReturn.BREAK:
                logging.verbose_debug(f"Jacobian ik break due to callback after {i} iter for q={q}.")
                break

            if cb_ret is CallbackReturn.FALSE:
                logging.verbose_debug(
                    f"Jacobian ik continues due to callback after {i} iter for q={q} returning false.")
                continue

            if eef_pose.valid(self.fk(q)):
                logging.verbose_debug(f"Jacobian ik terminated successfully after {i} iter for q={q}.")
                info.intermediate_result.q = q  # A new successful result is always considered the best result
                success = True
                break

            cost = ik_cost_function(self, q, eef_pose.nominal.in_world_coordinates())
            improvement = previous_cost - cost
            mean_improvement = (alpha_average * improvement) + (1 - alpha_average) * mean_improvement
            previous_cost = cost
            if mean_improvement < convergence_threshold:
                logging.debug(f"Jacobian ik break due to stagnating improvements after {i} iter for q={q}.")
                break
            if improvement < convergence_threshold:
                # Last resort: Allow movements of joints that have earlier been blocked due to reaching their limit
                joint_mask = info.joint_mask.copy()

            if cost < info.intermediate_result.cost:
                info.intermediate_result.q = q
                info.intermediate_result.cost = cost
        else:
            logging.verbose_debug(f"Jacobian ik not successful due to reaching the maximum number of iterations {i}")

        info.max_iter -= (i + 1)  # We start with an initial guess already
        return info.intermediate_result.q, success

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
        for i in range(1, self.dof + 1):
            # Move all joints that are directly connected to "world" (i.e. the first joint[s])
            if self.model.parents[i] != 0:
                continue
            try:
                self.model.jointPlacements[i] = displacement * self.model.jointPlacements[i]
            except IndexError:
                assert self.dof == 0, "Could not find the first joint, but there seems to be one"
        self.update_configuration(self.configuration)
        self._base_placement = Transformation(self.placement.multiply_from_left(displacement.homogeneous),
                                              set_safe=True)

    def random_configuration(self, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """
        Generates a random configuration.

        :param rng: Random number generator to use. If not given, this method uses the pinocchio default.
        """
        if rng is None:
            try:
                return pin.randomConfiguration(self.model)
            except RuntimeError:
                logging.debug('Random configuration failed due to infinite joint limits - setting to (-2pi, 2pi)')
        return super().random_configuration(rng)

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
        velocity_given = dq is not None
        acceleration_given = ddq is not None

        if acceleration_given and not velocity_given:
            raise ValueError("If ddq is specified, dq must be specified as well")

        q, dq, ddq = map(self._resize_for_pinocchio, (q, dq, ddq))
        self.configuration = q
        if velocity_given:
            self.velocities = dq
            if acceleration_given:
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
                           return_at_first: bool,
                           safety_margin: float = 0,
                           ) -> Union[bool, List[Tuple[RobotBase, Union[RobotBase, Obstacle]]]]:
        """
        Checks all collision pairs or all collisions pairs until the first colliding one is found.

        :param task: The task to check for collisions
        :param return_at_first: If true, this function behaves like a "has_collision". If false, it behaves like a
            "get_all_collisions" function.
        :param safety_margin: The safety margin is a padding added to all obstacles to increase needed separation to be
            considered collision free. In the same units as the collision objects (default and suggested in meters).
        :return: Bool if return_at_first is true, otherwise a list of all colliding pairs.
        """
        collisions: List[Tuple[RobotBase, Union[RobotBase, Obstacle]]] = list()
        request = CollisionRequest()
        request.security_margin = safety_margin
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
                    result.clear()
                    collisions.append((self, obstacle))
                    break  # Next obstacle
        if return_at_first:
            return False
        return collisions

    def collisions(self,
                   task: 'Task.Task',  # noqa: F821
                   safety_margin: float = 0,
                   ) -> List[Tuple[RobotBase, Union[RobotBase, Obstacle]]]:
        """
        Returns all collisions that appear between the robot and itself or obstacles in the task.

        :param task: The task to check for collisions
        :param safety_margin: The safety margin is a padding added to all obstacles to increase needed separation.
        :return: A list of tuples (robot, obstacle) for robot an obstacle in collision
        """
        return self.__check_collisions(task, return_at_first=False, safety_margin=safety_margin)

    def has_collisions(self, task: 'Task.Task', safety_margin: float = 0) -> bool:  # noqa: F821
        """
        Returns true if there is at least one collision in the task or one self-collision.

        :param task: The task the robot is in.
        :param safety_margin: The safety margin is a padding added to all obstacles to increase needed separation.
        :return: Boolean indicator whether there are any collisions.
        """
        return self.__check_collisions(task, return_at_first=True, safety_margin=safety_margin)

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
        kwargs.setdefault('joint_name', 'Joint_' + ''.join(self._rng.choice(list(string.ascii_letters), 5)))

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
