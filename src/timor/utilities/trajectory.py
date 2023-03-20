from __future__ import annotations

from dataclasses import dataclass, field
import string
from typing import Dict, Iterable, Optional, Union

import matplotlib.pyplot as plt
import meshcat
import numpy as np
from pinocchio.visualize import MeshcatVisualizer
from roboticstoolbox import mstraj

from timor import ModuleAssembly
from timor.utilities import logging
from timor.utilities.dtypes import Lazy
from timor.utilities.errors import TimeNotFoundError
from timor.utilities.jsonable import JSONable_mixin
from timor.utilities.tolerated_pose import ToleratedPose


@dataclass
class Trajectory(JSONable_mixin):
    """
    Holds an ordered set of states (can be any one of these):

    * robot configuration q + change
    * poses

    each with or without time information.
    """

    t: Optional[Union[np.ndarray, float]] = None
    q: Optional[np.ndarray] = None
    dq: np.ndarray = None
    _dq: Union[Lazy, np.ndarray] = field(init=False, repr=False)
    ddq: Optional[np.ndarray] = None
    _ddq: Union[Lazy, np.ndarray] = field(init=False, repr=False)
    pose: Optional[np.ndarray] = None
    goal2time: Dict[str, float] = field(default_factory=lambda: dict())
    # What time difference is considered the same time; should not need to change
    _allowed_deviation_in_time: float = 1e-12

    def __post_init__(self):
        """Check consistency such as same size for t, q, dq, ddq, and / or pose."""
        self.q = None if self.q is None else np.asarray(self.q)
        if self.has_q and len(self.q.shape) == 1:
            logging.info(f"Assuming q with single dimension is multiple steps (shape = {self.q.shape}); "
                         f"to force otherwise please give q as 1 x {self.q.shape[0]} array")
            self.q = self.q[:, None]
        self.pose = None if self.pose is None else np.asarray(self.pose)

        if not self.has_q and not self.has_poses:
            raise TypeError("Need at least pose or q.")

        if not self.has_q ^ self.has_poses:
            raise ValueError("q and pose are mutually exclusive")

        """Validity checks time"""
        if self.is_timed:
            self.t = np.asarray(self.t)

            if self.t.size == 1:  # Equidistant time steps
                if self.has_q:
                    if len(self.q.shape) == 2:  # Multi-step q
                        steps = self.q.shape[0]
                    elif len(self.q.shape) == 1:  # Trajectory with single step
                        steps = 1
                    else:
                        raise ValueError("Cannot extend t to all q.")
                elif self.pose.shape[0] >= 1:
                    steps = self.pose.shape[0]
                else:
                    raise ValueError("Cannot use sample size with neither q nor pose given.")
                if steps == 0:
                    self.t = np.zeros((0,), dtype=float)
                elif steps == 1:
                    self.t = np.array([self.t], dtype=float).reshape((1,))
                elif steps > 1:
                    self.t = np.arange(steps, dtype=float) * self.t
                else:
                    raise ValueError("Unsupported number of steps for a Trajectory: " + str(steps))
            else:
                if np.any(self.t[1:] - self.t[:-1] <= 0.):
                    raise ValueError("Expect time to be strictly increasing.")

            assert np.unique(self.t).size == self.t.size, "There cannot be duplicate time steps in a Trajectory"

            if any(t_goal < self.t[0] - self._allowed_deviation_in_time
                   or self.t[-1] + self._allowed_deviation_in_time < t_goal for t_goal in self.goal2time.values()):
                raise ValueError("Expect time of goal to be within time described by this trajectory.")
        else:
            if self.has_goals:
                raise ValueError("Goals needs time information in trajectory.")

        if self.has_q:
            if self.has_poses:
                raise ValueError("q and pose are mutually exclusive")

            if self.is_timed and self.q.shape[0] != self.t.shape[0]:
                raise ValueError("q and t need to be same length")

            if isinstance(self.dq, Iterable):
                self.dq = np.asarray(self.dq)
            elif self.is_timed:
                if len(self) > 1:
                    self._dq = Lazy(lambda: np.gradient(self.q, self.t, axis=0))
                    logging.debug("Created dq by numeric gradient.")
                else:
                    self._dq = Lazy(lambda: np.zeros_like(self.q))
            else:
                self._dq = None

            if isinstance(self.ddq, Iterable):
                self.ddq = np.asarray(self.ddq)
            elif self.is_timed:
                if len(self) > 1:
                    self._ddq = Lazy(lambda: np.gradient(self.dq, self.t, axis=0))
                    logging.debug("Created ddq by numeric gradient.")
                else:
                    self._ddq = Lazy(lambda: np.zeros_like(self.q))
            else:
                self._ddq = None

        if self.has_poses and self.is_timed and self.pose.shape != self.t.shape:
            raise ValueError("t and pose must be same length")

        if self._allowed_deviation_in_time < 0.:
            raise ValueError("Allowed Deviation in time should be >= 0. seconds")

    @classmethod
    def empty(cls) -> Trajectory:
        """Create and return an empty, timed, q-based trajectory"""
        return cls(t=np.zeros((0,)), q=np.zeros((0, 0)))

    @classmethod
    def from_json_data(cls, d: Dict, **kwargs) -> Trajectory:
        """Load from dict"""
        if "pose" in d:
            d["pose"] = np.asarray(tuple(ToleratedPose.from_json_data(P) for P in d.pop("pose")))
        return cls(**d)

    @classmethod
    def from_mstraj(cls, via_points: np.array, dt: float, t_acc: float, qd_max: float, **kwargs) -> Trajectory:
        """
        Create multi-segment multi-axis trajectory.

        See: https://petercorke.github.io/robotics-toolbox-python/arm_trajectory.html?highlight=mstraj

        :param via_points: Via points to connect with PTP trajectory. Start at first row, stop at last
        :param dt: sample time in seconds
        :param t_acc: acceleration time per PTP segment
        :param qd_max: maximum speed in each axis
        :param kwargs: Additional key word arguments piped directly into the mstraj function
        """
        traj = mstraj(via_points, dt, t_acc, qd_max, **kwargs)

        return cls(t=traj.t, q=traj.q, goal2time={})

    @classmethod
    def stationary(cls, time: float, *, q: np.array = None, pose: ToleratedPose = None) -> Trajectory:
        """
        Create a trajectory that stays in place for time seconds.

        :param time: Time period to stay at this configuration.
        :param q: Configuration to stay in (mutually exclusive with pose).
        :param pose: Tolerated pose to stay in (mutually exclusive with configuration).
        :return: The stationary trajectory, composed of two configurations / pose and a zero velocity / acceleration
        """
        if (pose is None) == (q is None):
            raise ValueError("Either q or pose required")
        t = np.asarray([0, time])
        if q is not None:
            if q.ndim > 1:
                raise ValueError("Stationary trajectory expects a single q")
            q = np.tile(q, (2, 1))
            dq = np.zeros_like(q)
            ddq = np.zeros_like(q)
            return cls(t=t, q=q, dq=dq, ddq=ddq, goal2time={})
        else:
            return cls(t=t, pose=np.asarray((pose, pose)))

    @property
    def dq(self) -> np.ndarray:
        """The joint velocities."""
        if isinstance(self._dq, Lazy):
            return self._dq()
        return self._dq

    @dq.setter
    def dq(self, dq: np.ndarray):
        if isinstance(dq, np.ndarray):
            if dq.shape != self.q.shape:
                raise ValueError(f"q and dq need same shape {dq.shape} != {self.q.shape}")
            if len(dq.shape) == 1:
                logging.info(f"Assuming dq with single dimension is multiple steps (shape = {dq.shape}); "
                             f"to force otherwise please give dq as 1 x {dq.shape[0]} array")
                dq = dq[:, None]
        self._dq = dq

    @property
    def ddq(self) -> np.ndarray:
        """The joint accelerations."""
        if isinstance(self._ddq, Lazy):
            return self._ddq()
        return self._ddq

    @ddq.setter
    def ddq(self, ddq: np.ndarray):
        if isinstance(ddq, np.ndarray):
            if ddq.shape != self.q.shape:
                raise ValueError(f"q and ddq need same shape {ddq.shape} != {self.q.shape}")
            if len(ddq.shape) == 1:
                logging.info(f"Assuming ddq with single dimension is multiple steps (shape = {ddq.shape}); "
                             f"to force otherwise please give dq as 1 x {ddq.shape[0]} array")
                ddq = ddq[:, None]
        self._ddq = ddq

    @property
    def dof(self):
        """Number of degrees of freedom encoded by this trajectory."""
        if not self.has_q:
            raise ValueError("DoF only defined for joint trajectory with q")
        return self.q.shape[1]

    @property
    def is_timed(self) -> bool:
        """Does this trajectory enforce time."""
        return self.t is not None

    @property
    def has_poses(self) -> bool:
        """Does this trajectory enforce a pose."""
        return self.pose is not None

    @property
    def has_q(self) -> bool:
        """Does this trajectory enforce a configuration"""
        return self.q is not None

    @property
    def has_dq(self) -> bool:
        """Does this trajectory enforce a configuration"""
        return self.has_q and self._dq is not None

    @property
    def has_ddq(self) -> bool:
        """Does this trajectory enforce a configuration"""
        return self.has_q and self._ddq is not None

    @property
    def has_goals(self) -> bool:
        """Does this trajectory refer to goal2time"""
        return len(self.goal2time) > 0

    def plot(self, show_goals_reached: bool = False, show_figure: bool = True) -> plt.Figure:
        """
        Plot this trajectory

        :param show_goals_reached: If set to true, the time at which goals are reached is highlighted in the plot
        :param show_figure: Toggle call to plt.show() which might disable some figure manipulations.
        :return: Figure with a plot of this trajectory.
        """
        if not self.has_q:
            raise NotImplementedError("Plot for trajectory with pose not yet implemented")

        if show_goals_reached and not self.is_timed:
            raise NotImplementedError("Cannot show goals if untimed trajectory")

        if self.is_timed:
            x = self.t
        else:
            x = range(len(self))

        f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex='all')
        ax1.plot(x, self.q)
        if self.has_dq:
            ax2.plot(x, self.dq)
        if self.has_ddq:
            ax3.plot(x, self.ddq)

        ax1.set(ylabel="Configuration q")
        ax2.set(ylabel="Velocity dq")
        if self.is_timed:
            ax3.set(xlabel="Time [s]", ylabel="Acceleration ddq")
        else:
            ax3.set(xlabel="Sample", ylabel="Acceleration ddq")

        if show_goals_reached:
            t_offset = 0.01 * (max(self.t) - min(self.t))
            for goal_id, goal_time in self.goal2time.items():
                for ax in (ax1, ax2, ax3):
                    ax.axvline(goal_time, c='blue')
                    ax.text(goal_time + t_offset, .9, goal_id, c='blue', transform=ax.get_xaxis_transform())

        plt.tight_layout()
        if show_figure:
            plt.show()

        return f

    def visualize(self, viz: MeshcatVisualizer, *,
                  name_prefix: Optional[str] = None,
                  scale: float = .33,
                  num_markers: Optional[int] = 2,
                  line_color: Optional[np.ndarray] = None,
                  assembly: Optional[ModuleAssembly] = None) -> None:
        """
        Visualize trajectory and add num_markers triads to show orientation.

        :param viz: The visualizer to show the trajectory in
        :param name_prefix: Name of the visualized trajectory.
        :param scale: size of the orientation markers.
        :param num_markers: how many orientation markers to show.
        :param line_color: An optional color (3x1 vector) for each pose (total len(self) x 3); default white
        :param assembly: An assembly to evaluate a joint space trajectory on; plots the end-effector movement
        """
        if self.has_poses:
            if name_prefix is None:
                name_prefix = f"traj-{''.join(np.random.choice(list(string.ascii_letters), 8))}"

            if num_markers > len(self):
                raise ValueError("Cannot draw more markers then there are pose in the trajectory.")
            for idx in np.linspace(0, len(self) - 1, num_markers, dtype=int):
                self.pose[idx].visualize(viz, name=f"traj_{name_prefix}_{idx}", scale=scale)

            line_segments = np.asarray([p.nominal.translation for ps in zip(self.pose[:-1], self.pose[1:])
                                        for p in ps], dtype=np.float32).T

        elif self.has_q:
            if assembly is None:
                raise ValueError("Can only visualize joint trajectory with an assembly.")

            line_points = np.asarray([assembly.robot.fk(q).translation for q in self.q])
            line_segments = np.empty((2 * (line_points.shape[0] - 1), line_points.shape[1]), dtype=line_points.dtype)
            line_segments[0::2, :] = line_points[:-1, :]
            line_segments[1::2, :] = line_points[1:, :]
            line_segments = line_segments.T

        else:
            raise NotImplementedError("Cannot visualize trajectory without q or poses.")

        if line_color is None:
            line_color = np.ones(line_segments.shape)
        line = meshcat.geometry.LineSegments(
            geometry=meshcat.geometry.PointsGeometry(position=line_segments, color=line_color),
            material=meshcat.geometry.LineBasicMaterial(vertexColors=True)
        )

        viz.viewer[f"traj_{name_prefix}_position"].set_object(line)

    def get_at_time(self, time: float) -> Trajectory:
        """Returns state at desired time."""
        if not self.is_timed:
            raise ValueError("Non-timed trajectory cannot access via time")

        time_idx = np.argmin(np.abs(self.t - time))
        if abs(self.t[time_idx] - time) > self._allowed_deviation_in_time:
            raise TimeNotFoundError(f"Could not find index for time {time} in timed Trajectory.")
        return self[time_idx]

    def to_json_data(self) -> Dict:
        """Transform into a jsonable dict."""
        ret = {}
        for k in ("t", "q", "dq", "ddq"):
            if hasattr(self, k) and isinstance(getattr(self, k), np.ndarray):
                ret[k] = getattr(self, k).tolist()
        ret["goal2time"] = self.goal2time
        if self.has_poses:
            ret["pose"] = tuple(p.to_json_data() for p in self.pose)
        for k, v in self.__dict__.items():
            assert k in ret.keys() or k[0] == "_" or v is None or len(v) == 0, \
                "Serialization should mention all non-None, non-private parts"

        if isinstance(self._dq, Lazy):
            ret.pop("dq", None)
        if isinstance(self._ddq, Lazy):
            ret.pop("ddq", None)

        return ret

    def __add__(self, other) -> Trajectory:
        """Appends other trajectory to self."""
        if not isinstance(other, Trajectory):
            return NotImplemented

        if self.has_q != other.has_q:
            raise ArithmeticError("Cannot concatenate q and pose trajectories")

        if self.is_timed != other.is_timed:
            raise ArithmeticError("Cannot concatenate timed and untimed trajectories")

        if len(other) == 0:  # Short circuit adding with empty trajectory
            return self
        if len(self) == 0:
            return other

        q_new = dq_new = ddq_new = t_new = None
        poses_new = None
        goals_new = {}
        if self.is_timed:
            if len(self) > 1:
                # dt is the offset between end of self and start of other
                dt = self.t[-1] - self.t[-2]
                logging.info(f"Calculated step between added trajectories as {dt} from left hand trajectory")
            else:
                if len(other) < 2:
                    raise ValueError("Adding trajectories needs to deduce step between trajectories "
                                     "and therefore needs at least one with length > 1")
                dt = other.t[-1] - other.t[-2]
                logging.info(f"Calculated step between added trajectories as {dt} from right hand trajectory")

            offset_other = self.t[-1] + dt
            t_new = np.hstack((self.t, other.t + offset_other))
            goals_new = {**self.goal2time,
                         **{k: v + offset_other for k, v in other.goal2time.items()}}  # Offset goal times

        if self.has_q:
            q_new = np.vstack((self.q, other.q))
            if isinstance(self._dq, np.ndarray) or isinstance(other._dq, np.ndarray):
                dq_new = np.vstack((self.dq, other.dq))
            else:
                dq_new = None
            if isinstance(self._ddq, np.ndarray) or isinstance(other._dq, np.ndarray):
                ddq_new = np.vstack((self.ddq, other.ddq))
            else:
                ddq_new = None

        elif self.has_poses:
            poses_new = np.concatenate((self.pose, other.pose))

        return Trajectory(t=t_new, q=q_new, goal2time=goals_new, dq=dq_new, ddq=ddq_new, pose=poses_new)

    def __eq__(self, other) -> bool:
        """Check if two trajectories are equal."""
        if isinstance(other, self.__class__):
            return len(self) == len(other) \
                and self.has_q == other.has_q \
                and self.has_poses == other.has_poses \
                and self.is_timed == other.is_timed \
                and (np.allclose(self.t, other.t) if self.is_timed else True) \
                and (np.allclose(self.q, other.q) if self.has_q else True) \
                and (np.allclose(self.dq, other.dq) if self.has_dq else True) \
                and (np.allclose(self.ddq, other.ddq) if self.has_ddq else True) \
                and (all(p_self == p_other for p_self, p_other in zip(self.pose, other.pose))
                     if self.has_poses else True)\
                and self.goal2time.keys() == other.goal2time.keys() \
                and all(abs(self.goal2time[k] - other.goal2time[k]) < 1e-5 for k in self.goal2time.keys())
        return NotImplemented

    def __getitem__(self, item: Union[int, Iterable[int], slice]) -> Trajectory:
        """Get sub-trajectory via indexing."""
        if not isinstance(item, slice):
            if np.isscalar(item):
                item = np.asarray((item,))  # Internally need 2D views on q, etc. -> unpack for external
            else:
                item = np.asarray(item)
        if self.is_timed:
            ts = self.t[item]
            t_min, t_max = np.min(ts), np.max(ts)  # Calculate time range to select goals with
        return self.__class__(
            t=(self.t[item] if self.is_timed else None),
            q=(self.q[item, :] if self.has_q else None),
            dq=(self.dq[item, :] if self.has_dq else None),
            ddq=(self.ddq[item, :] if self.has_ddq else None),
            goal2time={goal_id: t_g for goal_id, t_g in self.goal2time.items() if (t_min <= t_g <= t_max)},
            pose=(self.pose[item] if self.has_poses else None)
        )

    def __iter__(self):
        """Explicit iterator over trajectory."""
        return (self[i] for i in range(len(self)))

    def __len__(self) -> int:
        """Return number of samples in this trajectory"""
        if self.is_timed:
            return self.t.shape[0]
        if self.has_q:
            return self.q.shape[0]
        if self.has_poses:
            return len(self.pose)
        raise ValueError("No length determining part set for this trajectory")

    def __getstate__(self):
        """Custom get state, e.g. for pickle / deepcopy."""
        return self.to_json_data()

    def __setstate__(self, state: Dict):
        """Custom set state, e.g. for pickle / deepcopy."""
        cpy = self.__class__.from_json_data(state)
        self.__dict__ = cpy.__dict__
