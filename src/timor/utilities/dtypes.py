from __future__ import annotations

from collections import namedtuple
from dataclasses import asdict, dataclass, field, fields
import datetime
import logging
from pathlib import Path
import random
import re
from typing import Any, Callable, Collection, Dict, Generator, Generic, Iterable, List, Tuple, Type, TypeVar, Union, \
    get_type_hints

from hppfcl import hppfcl
from matplotlib import pyplot as plt
import numpy as np
import pinocchio as pin
from roboticstoolbox.tools.trajectory import mstraj

import timor.utilities.errors as err
from timor.utilities.schema import DEFAULT_DATE_FORMAT
from timor.utilities.transformation import Transformation, TransformationLike

T = TypeVar('T')
IntermediateIkResult = namedtuple("IntermediateIkResult", ["q", "distance"])


class EternalDict(dict):
    """An immutable dictionary that prevents changing any of its items, but allows adding new ones"""

    def __hash__(self):
        """Custom hash function"""
        return id(self)

    def __setitem__(self, key, value):
        """The only way possible to change this dictionary: Add new values"""
        if key in self:
            self._immutable()
        super().__setitem__(key, value)

    def _immutable(self, *args, **kws):
        raise TypeError('Object is supposedly set immutable (Eternal Dict)')

    __delitem__ = _immutable
    clear = _immutable
    update = _immutable
    pop = _immutable
    popitem = _immutable


class Lazy(Generic[T]):
    """An implementation of lazy variables in python"""

    def __init__(self, func: Callable[[], T]):
        """Initialize an "empty" lazy variable with the function to be evaluated lazily"""
        self.func: Callable[[], T] = func
        self.value = None

    def __call__(self, *args, **kwargs) -> T:
        """Useful s.t. the variable can be evaluated like lazy()"""
        return self.value_or_func()

    def __mul__(self, other):
        """Implemented to prevent evaluation of Lazy variables when multiplied with another lazy variable"""
        if isinstance(other, Lazy):
            def chain():
                return self.value_or_func() * other.value_or_func()

            return Lazy(chain)
        else:
            return self() * other

    def __rmul__(self, other):
        """Implemented to prevent evaluation of Lazy variables when multiplied with another lazy variable"""
        if isinstance(other, Lazy):
            def chain():
                return other.value_or_func() * self.value_or_func()

            return Lazy(chain)
        else:
            return other * self()

    def __getstate__(self) -> T:
        """Lazy variables will always be evaluated when serialized"""
        return self.value_or_func()

    def reset(self):
        """Reset the value of the lazy variable s.t. it must be re-evaluated next time the value is requested"""
        self.value = None

    def value_or_func(self) -> T:
        """Evaluates the lazy variable. Is triggered by self()"""
        if self.value is None:
            self.value = self.func()
        return self.value

    @classmethod
    def from_variable(cls, variable: any) -> Lazy:
        """Useful wrapper to prevent evaluation of Lazy variables when multiplied with another variable"""
        def ret():
            """Utility function"""
            return variable

        lazy_variable = cls(ret)
        lazy_variable.value = variable
        return lazy_variable


class SingleSet(set):
    """
    A set that does not accept duplicates silently.

    Like a set, but instead of silently ignoring duplicates, it throws an error whenever an element that is
    already in this set is added.
    Inspired by: https://stackoverflow.com/questions/41281346/how-to-raise-error-if-user-tries-to-enter-duplicate-entries-in-a-set-in-python  # noqa: E501
    Only for union it is made sure that a SingleSet is returned. Diff, intersection, ... methods will return basic sets!
    """

    _err = "Element {} already present!"

    def __init__(self, *elements: Iterable):
        """
        Custom init function.

        Might be slow, but provides safety that every element in the input argument is checked using the tested update
        and add methods.
        For sophisticated element checking, consider overriding the __contains__ method in sub-classes.
        """
        super().__init__()
        self.update(*elements)

    def add(self, element) -> None:
        """Custom add to set needed s.t. uniqueness is ensured"""
        if element in self:
            raise err.UniqueValueError(self._err.format(element))
        super().add(element)

    def copy(self) -> SingleSet:
        """Custom copy should also return a SingleSet"""
        return self.__class__(super().copy())

    def filter(self, func: Callable[[any], bool]) -> SingleSet:
        """
        Convenience function to avoid set comparison over and over again

        :param func: A filter function that takes an element of this set and evaluates to true if
            the element should be kept
        :return: A new set of elements fulfilling the filter criterion
        """
        return self.__class__(element for element in self if func(element))

    def intersection(self, *s: Iterable) -> SingleSet:
        """Intersection should also retutrn a SingleSet"""
        return self.__class__(element for arg in s for element in arg if element in self)

    def update(self, *s: Iterable) -> None:
        """Update method that does not change this instance if the input iterable contains duplicates"""
        added = set()
        for arg in s:
            for element in arg:
                if element in self:
                    for previously_added in added:  # Reset
                        self.remove(previously_added)

                    raise err.UniqueValueError(self._err.format(element))
                self.add(element)
                added.add(element)

    def union(self, *s: Iterable) -> SingleSet:
        """Returns a SingleSet of the union of this and the input"""
        elements = self.__class__()  # This is needed so this classes __contains__ method is used on the input as well
        for arg in s:
            for element in arg:
                if element in self:
                    raise err.UniqueValueError(self._err.format(element))
                elif element in elements:
                    raise err.UniqueValueError(f"The element {element} is provided multiple times in the input.")
                elements.add(element)
        return self.__class__(super().union(elements))


@dataclass
class TypedHeader:
    """A dataclass that ensures and transforms {Module, Task, Solution}-header data to the desired datatypes."""

    def __post_init__(self):
        """Post init function that ensures the correct datatypes"""
        self.__setattr__('__immutable', False)
        for name, type_annotation in get_type_hints(self, globalns=globals(), localns=locals()).items():
            value = self.__getattribute__(name)
            self.__setattr__(name, self.cast(value, type_annotation))
        self.__setattr__('__immutable', True)

    @staticmethod
    def cast(value: Any, cast_to_type: Type) -> Any:
        """Casts the value to the dtype"""
        if cast_to_type is datetime.datetime:
            return TypedHeader.cast_to_datetime(value)
        try:
            return cast_to_type(value)
        except TypeError:
            logging.debug("Could not cast {} to type {}. Returning original value.".format(value, cast_to_type))
            return value

    @staticmethod
    def cast_to_datetime(value: Union[str, datetime.datetime]) -> datetime.datetime:
        """Casts the value to a datetime.datetime object"""
        if isinstance(value, datetime.datetime):
            return value
        if isinstance(value, str):
            value = re.sub('/', '-', value)  # Allows parsing yyyy/mm/dd formatted dates
            return datetime.datetime.strptime(value, DEFAULT_DATE_FORMAT)
        raise TypeError(f"Cannot cast {value} to datetime.datetime")

    @staticmethod
    def string_list_factory() -> field:
        """Returns a field with a default factory that returns a list with one empty string"""

        def list_factory() -> List[str]: return ['']

        return field(default_factory=list_factory)

    @classmethod
    def fields(cls) -> Tuple[str, ...]:
        """Returns the fields of this class"""
        return tuple(f.name for f in fields(cls))

    def asdict(self) -> Dict[str, Any]:
        """Returns a dictionary representation of this object"""
        return asdict(self)

    def _raise_immutable(self, f: Callable) -> Callable:
        """Raises error if objects in this dataclass are to be resetted but the instance is set to immutable"""

        def wrapper(*args, **kwargs):
            try:
                if self.__getattribute__('__immutable'):
                    raise TypeError("Cannot reset attributes of an immutable object")
            except AttributeError:  # __init__, __immutable not yet set
                pass
            return f(*args, **kwargs)

        return wrapper

    def __delattr__(self, item):
        """Delete attribute"""
        self._raise_immutable(super().__delattr__)(item)

    def __setattr__(self, key, value):
        """Sets an attribute of this class"""
        return self._raise_immutable(super().__setattr__)(key, value)


@dataclass
class TorqueInput:
    """Holds data defining robot control parameters"""

    t: Union[np.ndarray, float]
    torque: np.ndarray
    goals: Dict[str, float]  # Maps goals to the point in time they are achieved in the trajectory

    def __post_init__(self):
        """Sanity checks after dataclass values are written"""
        if isinstance(self.t, (int, float)):  # Equidistant time steps
            steps = self.torque.shape[0]
            self.t = np.arange(steps * self.t, dtype=float)

        assert len(np.unique(self.t)) == len(self.t), "There cannot be duplicate time steps in a Torque Input"


@dataclass
class Trajectory:
    """Holds trajectory data and makes sure that input arguments are valid"""

    t: Union[np.ndarray, float]
    q: np.ndarray
    goals: Dict[str, float] = field(default_factory=lambda: dict())  # Maps goals to the point in time they are achieved
    dq: np.ndarray = None
    ddq: np.ndarray = None

    def __post_init__(self):
        """Enforce typing"""
        self.t = np.asarray(self.t)
        self.q = np.asarray(self.q)

        """Validity checks"""
        if self.t.size == 1:  # Equidistant time steps
            if len(self.q.shape) == 2:
                steps = self.q.shape[0]
                self.t = np.arange(steps, dtype=float) * self.t

        assert np.unique(self.t).size == self.t.size, "There cannot be duplicate time steps in a Trajectory"

        if (self.dq is None) or (self.ddq is None):
            try:
                time_delta = self.t[1:] - self.t[:-1]
                init = np.zeros((1, self.q.shape[1]), dtype=float)
            except IndexError as exe:
                if len(self.q.shape) == 1:
                    init = np.zeros((self.q.shape[0],), dtype=float)
                    self.dq = init
                    self.ddq = init
                else:
                    raise exe

        if self.dq is not None:
            self.dq = np.asarray(self.dq)
            assert self.q.shape == self.dq.shape, \
                f"Invalid trajectory, shape of q {self.q.shape} and dq {self.dq.shape} do not match."
        else:
            # Approximate velocity TODO : Use numpy gradient?
            dq = (self.q[1:] - self.q[:-1]) / time_delta[:, None]
            self.dq = np.vstack([init, dq])

        if self.ddq is not None:
            self.ddq = np.asarray(self.ddq)
            assert self.q.shape == self.ddq.shape, \
                f"Invalid trajectory, shape of q {self.q.shape} and ddq {self.ddq.shape} do not match."
        else:
            # Approximate acceleration TODO : Use numpy gradient?
            ddq = (self.dq[1:] - self.dq[:-1]) / time_delta[:, None]
            self.ddq = np.vstack([init, ddq])

    @staticmethod
    def from_mstraj(via_points: np.array, dt: float, t_acc: float, qd_max: float, **kwargs) -> 'Trajectory':
        """
        Create multi-segment multi-axis trajectory.

        See: https://petercorke.github.io/robotics-toolbox-python/arm_trajectory.html?highlight=mstraj

        :param via_points: Via points to connect with PTP trajectory. Start at first row, stop at last
        :param dt: sample time in seconds
        :param t_acc: acceleration time per PTP segment
        :param qd_max: maximum speed in each axis
        :param kwargs: Additional key word arguments piped directly into the mstraj function
        """
        t = mstraj(via_points, dt, t_acc, qd_max, **kwargs)
        dq = np.gradient(t.q, axis=0)
        ddq = np.gradient(dq, axis=0)

        return Trajectory(t=t.t, q=t.q, dq=dq, ddq=ddq, goals={})

    @staticmethod
    def stationary(configuration: np.array, time: float) -> 'Trajectory':
        """
        Create a trajectory that stays in place for time seconds.

        :param configuration: Configuration to stay in.
        :param time: Time period to stay at this configuration.
        :return: The stationary trajectory, composed of two configurations and a zero velocity / acceleration
        """
        q = np.tile(configuration, (2, 1))
        dq = np.zeros_like(q)
        ddq = np.zeros_like(q)
        return Trajectory(t=np.asarray([0, time]), q=q, dq=dq, ddq=ddq, goals={})

    def to_json_data(self) -> Dict:
        """Return trajectory as a dictionary that can be serialized as json."""
        return {
            "t": self.t.tolist(),
            "q": self.q.tolist(),
            "dq": self.dq.tolist(),
            "ddq": self.ddq.tolist(),
            "goal2time": self.goals
        }

    def plot(self, show_goals_reached: bool = False) -> plt.Figure:
        """
        Plot this trajectory

        :param show_goals_reached: If set to true, the time at which goals are reached is highlighted in the plot
        :return: Figure with a plot of this trajectory.
        """
        f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex='all')
        ax1.plot(self.t, self.q)
        ax2.plot(self.t, self.dq)
        ax3.plot(self.t, self.ddq)

        ax1.set(ylabel="Configuration q")
        ax2.set(ylabel="Velocity dq")
        ax3.set(xlabel="Time [s]", ylabel="Acceleration ddq")

        if show_goals_reached:
            t_offset = 0.01 * (max(self.t) - min(self.t))
            for goal_id, goal_time in self.goals.items():
                for ax in (ax1, ax2, ax3):
                    ax.axvline(goal_time, c='blue')
                    ax.text(goal_time + t_offset, .9, goal_id, c='blue', transform=ax.get_xaxis_transform())

        plt.tight_layout()
        plt.show()

        return f

    def __add__(self, other: 'Trajectory'):
        """Append another trajectory to self"""
        if not isinstance(other, Trajectory):
            return NotImplemented
        if (other.t.size == 0) or (self.t.size == 0):
            return self
        elif self.t.size > 1:
            # dt is the offset between end of self and start of other
            dt = self.t[-1] - self.t[-2]
        else:
            if other.t.size < 2:
                raise ValueError("Adding trajectories needs to deduce step between trajectories "
                                 "and therefore needs at least one with length > 1")
            dt = other.t[-1] - other.t[-2]

        offset_other = dt + self.t[-1]
        t_new = np.hstack((self.t, other.t + offset_other))
        q_new = np.vstack((self.q, other.q))
        # Offset the second trajectories goals
        goals_new = {**self.goals, **{k: v + offset_other for k, v in other.goals.items()}}

        dq_new = np.vstack((self.dq, other.dq))
        ddq_new = np.vstack((self.ddq, other.ddq))

        return Trajectory(t=t_new, q=q_new, goals=goals_new, dq=dq_new, ddq=ddq_new)

    def __getitem__(self, item) -> Trajectory:
        """Allows to index a trajectory like one would index every single item of the trajectory"""
        return self.__class__(
            t=self.t[item],
            q=self.q[item],
            goals={k: v for k, v in self.goals.items() if (np.min(self.t[item]) <= v and v <= np.max(self.t[item]))},
            dq=self.dq[item],
            ddq=self.ddq[item]
        )

    def __eq__(self, other) -> bool:
        """Test if two trajectories are close together (max distance < 1e-5)"""
        if type(self) is type(other):
            return all((np.allclose(self.t, other.t),
                        np.allclose(self.q, other.q),
                        np.allclose(self.dq, other.dq),
                        np.allclose(self.ddq, other.ddq),
                        self.goals.keys() == other.goals.keys(),
                        all(abs(self.goals[k] - other.goals[k]) < 1e-5 for k in self.goals.keys())))
        elif isinstance(other, Trajectory):
            raise NotImplementedError(f"Comparison of {type(self)} and {type(other)} not yet implemented.")
        return NotImplemented


def deep_iter_dict(d: Dict[any, any], max_depth=10, _prev=None) -> Generator[Tuple[List[any], any], None, None]:
    """
    Iterators over hierarchical dictionaries

    :param d: Any dictionary
    :param max_depth: The max recursion depth to dive into the dictionary
    :param _prev: Previously seen keys. Usually would not be set by the user.
    :return: Yields a generator that returns a tuple of:

        * The keys leading to this values in a list like: [key_lvl_1, key_lvl_2, ..., key_lvl_n]
        * The value if it is not yet another dictionary
    """
    if _prev is None:
        _prev = []
    if max_depth == 0:
        raise RecursionError("Dictionary contains more than the max. number of nested layers.")
    for key, val in d.items():
        if not isinstance(val, dict):
            yield _prev + [key], val
        else:
            yield from deep_iter_dict(val, max_depth - 1, _prev + [key])


def float2array(scalar: float) -> np.ndarray:
    """
    Handle sensitive bindings between C++ eigen and numpy.

    Methods like model.addJoint only accept np ndarrays of shape (1,) for inputs that originally are eigen matrices in
    C++.

    :param scalar: Could be any scalar really
    :return: A np array of shape (1,) containing this scalar (as float)
    """
    # Allow inputs that are already in the right format
    if isinstance(scalar, np.ndarray):
        assert scalar.size == 1, "float2array takes only one float at a time"
        scalar = scalar.flat[0]

    return np.asarray([scalar], float)


def fuzzy_dict_key_matching(dict_in: Dict[str, any],
                            aliases: Dict[str, str] = None,
                            desired_only: Tuple = None,
                            ignore_casing: bool = True) -> Dict[str, any]:
    """
    Fuzzy dictionary key matching.

    This method provides a helper utility to find keys in an input dictionary that are not exactly right,
    but are an alias for an expected key or differ only by casing.

    :param dict_in: An input dictionary that shall be "fuzzy-matched". The keys must be strings.
    :param aliases: A mapping from "valid alias key in dict_in" -> "valid real key"
    :param desired_only: If given, the returned dictionary will only contain this subset of keys.
        Useful for discarding aliases after matching
    :param ignore_casing: If true, keys will be matched case-insensitive to aliases AND TO KEYS FROM DESIRED ONLY
    :return: The resulting dictionary containing all or only the desired mappings, taking aliases into account
    """
    if aliases is None:
        aliases = {}
    if ignore_casing:
        # Define the operation performed on
        def strfnc(s: str):
            return s.lower()
    else:
        def strfnc(s: str):
            pass
    if desired_only is not None:
        # Add them to aliases to possibly enable string-insensitive matching for desired keys
        aliases.update({desired: desired for desired in desired_only})
        cpy = aliases.copy()
        for key in cpy:
            # Sometimes an alias might be provided, even if it should not be returned
            if aliases[key] not in desired_only:
                del aliases[key]

    comp_keys = [strfnc(key) for key in dict_in]
    return_dict = dict_in if desired_only is None else {k: dict_in[k] for k in desired_only if k in dict_in}
    for alias_key, valid_key in aliases.items():
        if valid_key not in dict_in:  # Is the "real" value already present?
            if strfnc(alias_key) in comp_keys:  # Is there an alias to match?
                # Find the one matching value
                matching = [val for existing_key, val in dict_in.items() if strfnc(existing_key) == strfnc(alias_key)]
                assert len(matching) == 1, "Multiple similar keys for {} found in header".format(valid_key)
                return_dict[valid_key] = matching[0]

    return return_dict


def hppfcl_collision2pin_geometry(geometry: hppfcl.CollisionGeometry,
                                  transform: TransformationLike,
                                  name: str,
                                  parent_joint: int,
                                  parent_frame: int = None
                                  ) -> pin.GeometryObject:
    """Creates a pinocchio geometry object as used for robot bodies from a hppfcl collision object."""
    placement = pin.SE3(Transformation(transform).homogeneous)
    if parent_frame is None:
        return pin.GeometryObject(name, parent_joint, geometry, placement)
    else:
        return pin.GeometryObject(name, parent_frame, parent_joint, geometry, placement)


def map2path(p: Union[str, Path]) -> Path:
    """Maps strings to paths, but does nothing else s.t. e.g. Django paths are not casted."""
    if isinstance(p, str):
        return Path(p)
    return p


def possibly_nest_as_list(value: any, tuple_ok: bool = True):
    """
    A utility to wrap a value within a list if it is not already.

    Useful for loading jsons where arrays sometimes get unpacked if they contain only one element - which silently
    causes errors later in the pipeline.

    :param value: Basically any python variable
    :param tuple_ok: If true, tuples will not be packed as a list as their behavior in python is very similar.
    :return: [value] if value was not a list (or possibly a tuple) before, else value
    """
    if not isinstance(value, list):
        if tuple_ok and isinstance(value, tuple):
            return value
        else:
            return [value]

    # Value was a list already
    return value


def randomly(seq: Collection) -> Iterable:
    """
    Returns a random iterator over seq.

    :source: https://stackoverflow.com/questions/9252373/random-iteration-in-python
    """
    shuffled = list(seq)
    random.shuffle(shuffled)
    return iter(shuffled)


def uniform_collision_geometry(geometry: Union[hppfcl.CollisionObject, pin.pinocchio_pywrap.GeometryObject]) -> \
        Tuple[hppfcl.CollisionObject, Union[hppfcl.Transform3f, pin.SE3]]:
    """
    Takes either a hppfcl collision object or a pin geometry object and returns the according geometry + transform.

    This is needed if hppfcl collide() should be used on a combination of hhpfcl native
      and pin native collision geometries

    :return: A tuple of (geometry, transform) to feed in collide()
    """
    if isinstance(geometry, hppfcl.CollisionObject):
        return geometry.collisionGeometry(), geometry.getTransform()
    elif isinstance(geometry, pin.pinocchio_pywrap.GeometryObject):
        return geometry.geometry, geometry.placement
    raise ValueError("Unknown Geometry Type: {}".format(type(geometry)))
