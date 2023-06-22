#!/usr/bin/env python3
# Author: Jonathan Külz
# Date: 17.01.22
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import datetime
import itertools
import os
from pathlib import Path
from typing import Collection, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import uuid
import warnings

from hppfcl import hppfcl
import numpy as np
import pinocchio as pin

from timor.Robot import PinRobot, RobotBase
from timor.task import Constraints, Goals
from timor.task.Obstacle import Obstacle
from timor.utilities import configurations
from timor.utilities.dtypes import TypedHeader
from timor.utilities.file_locations import map2path, schema_dir
from timor.utilities.jsonable import JSONable_mixin
import timor.utilities.logging as logging
from timor.utilities.schema import DEFAULT_DATE_FORMAT, get_schema_validator
from timor.utilities.visualization import center_camera

GRAVITY = np.array([0, 0, -9.81], dtype=float)  # Default gravity for z upwards TODO : Move to global configuration?!
robot_or_robots = Union[PinRobot, Iterable[PinRobot]]


@dataclass
class TaskHeader(TypedHeader):
    """The header every task contains"""

    ID: str
    version: str = "2023"
    taskName: str = ""
    tags: List = field(default_factory=list)
    date: datetime.date = datetime.date.today()
    timeStepSize: float = .01  # 10ms
    gravity: np.ndarray = field(default_factory=GRAVITY.copy)
    author: List[str] = TypedHeader.string_list_factory()
    email: List[str] = TypedHeader.string_list_factory()
    affiliation: List[str] = TypedHeader.string_list_factory()

    @classmethod
    def empty(cls) -> TaskHeader:
        """Returns an empty task header"""
        return cls(ID=f'tmp_{uuid.uuid4()}')

    def __eq__(self, other):
        """Equality of headers is given by equality of attributes"""
        if not isinstance(other, TaskHeader):
            return NotImplemented
        return all(getattr(self, attr) == getattr(other, attr) for attr in self.__annotations__.keys())


class Task(JSONable_mixin):
    """A task describes a problem that must be solved by one or multiple robots.

    The problem can be composed of multiple goals, each with their own constraints.
    Global constraints, e.g. valid base placements or the order of goals can be defined on the task-level.
    """

    def __init__(self,
                 header: Union[TaskHeader, Dict],
                 obstacles: Collection[Obstacle] = None,
                 goals: Sequence['Goals.GoalBase'] = None,
                 constraints: Collection['Constraints.ConstraintBase'] = None
                 ):
        """Create a Task:

        :param header: The header contains meta informaton about the task. Must at least contain an ID.
        :param obstacles: Obstacles in the task/environment the robots should not collide with.
        :param goals: The complete set of goals that need to be achieved in order to solve the task
        :param constraints: "Global" constraint that must hold for the whole task.
        """
        self.header = header if isinstance(header, TaskHeader) else TaskHeader(**header)

        self.obstacles: List[Obstacle] = list()  # High-Level Objects
        self.goals: Tuple['Goals.GoalBase'] = tuple() if goals is None else tuple(goals)
        self.constraints: Tuple['Constraints.ConstraintBase'] = tuple() if constraints is None \
            else tuple(constraints)

        if obstacles is not None:
            for obstacle in obstacles:
                self.add_obstacle(obstacle)

    @classmethod
    def from_id(cls, id: str, package_dir: Path) -> Task:
        """
        Load a task based on its id stored in package_dir/id.json.

        :param id: id (including namespace with "/" to look for.
        :param package_dir: The directory to look for the task in.
        """
        return cls.from_json_file(package_dir.joinpath(f"{id}.json"))

    @staticmethod
    def _deduce_package_dir(filepath: Path, content: Union[Dict, List]) -> Path:
        """
        Deduce the package directory from the filepath and the content of the task.

        A task is always stored in <package_dir>/<task.id>.json. The package_dir is the directory; the task.id may also
        contain subdirectories / namespacing denoted with "/".

        :param filepath: The path to the task file
        :param content: The content of the task file
        return: The package directory that all file keys in content are relative to.
        """
        if not isinstance(content, dict):
            raise ValueError(f"Content of {filepath} describing task is not a dictionary.")
        if 'header' not in content or 'ID' not in content['header']:
            raise ValueError(f"Content of {filepath} describing task does not contain an ID.")
        id = content['header']['ID']
        if not filepath.match(f"**/{id}.json"):
            raise ValueError(f"Filepath {filepath} of task does not end in ID {id}")
        # Only keep parts of abs path before ID
        return Path(*filepath.with_suffix("").parts[:-len(Path(id).parts)])

    @classmethod
    def from_json_data(cls, d: Dict[str, any], *args, **kwargs):
        """
        Loads a task from a parsed json (=dictionary).

        :param d: The parsed json data
        """
        _, validator = get_schema_validator(schema_dir.joinpath("TaskSchema.json"))
        if not validator.is_valid(d):
            logging.debug(f"Errors: {tuple(validator.iter_errors(d))}")
            raise ValueError("task.json invalid.")
        content = deepcopy(d)  # Make sure we don't modify the original data while popping items
        header = TaskHeader(**{key: arg for key, arg in content.pop('header').items()})
        obstacles = [Obstacle.from_json_data(specs) for specs in content.pop('obstacles')]
        goals = [Goals.GoalBase.goal_from_json_data(goal) for goal in content.pop('goals', [])]
        if 'Constraints' in content:
            logging.error("Constraints is written in upper case but should be lower case!")
            content['constraints'] = content.pop('Constraints')
        constraints = [Constraints.ConstraintBase.from_json_data(c) for c in content.pop('constraints', [])]
        if len(content) > 0:
            raise ValueError("Unresolved keys: {}".format(', '.join(content)))
        task = cls(header, obstacles, goals=goals, constraints=constraints)
        return task

    def to_json_file(self, save_at: Union[Path, str],
                     handle_missing_assets: Optional[str] = None, *args, **kwargs):
        """
        Save the task to a json file.

        :param save_at: The path to save the task to.
          If it is a directory, the task will be saved to save_at/<task.id>.json.
          If it is a file, the task will be saved to save_at and it is ensured that the filename ends in <task.id>.json.
        :param handle_missing_assets: How to handle missing assets.
          See :meth:`timor.utilities.helper.handle_assets` for details.
        """
        if handle_missing_assets is None:
            handle_missing_assets = configurations.task_assets_copy_behavior
        save_at = map2path(save_at)
        if len(save_at.suffixes) > 0:  # there is a suffix -> this is a file; is_file only true if already exists
            if not save_at.match(f"**/{self.id}.json"):
                raise ValueError(f"Filepath {save_at} does not end in ID {self.id} of self.")
            new_package_dir = map2path(*save_at.with_suffix("").parts[:len(Path(self.id).parts)])
        elif save_at.is_dir():
            new_package_dir = save_at
            save_at = new_package_dir / f"{self.id}.json"
        else:
            raise ValueError("save_at must be a file or directory.")
        os.makedirs(save_at.parent, exist_ok=True)

        super().to_json_file(save_at, new_package_dir=new_package_dir,
                             handle_missing_assets=handle_missing_assets, *args, **kwargs)

    @staticmethod
    def from_srf_file(filepath: Path):
        """Future Work"""
        raise NotImplementedError()  # TODO

    @classmethod
    def empty(cls) -> Task:
        """Returns an empty task with an empty header"""
        return cls(header=TaskHeader.empty())

    def add_obstacle(self, obstacle: Obstacle):
        """
        Adds an obstacle to the existing task.

        :param obstacle: Any obstacle derived from the Obstacle class.
            This can contain a single geometry or compositions of such.
        """
        if obstacle.id in self.obstacle_names:
            raise ValueError("An obstacle with the ID {} already exists in the Task.".format(obstacle.id))
        self.obstacles.append(obstacle)

    def to_json_data(self) -> Dict[str, any]:
        """Returns the serialized task dictionary"""
        content = dict()

        # Header
        content['header'] = self.header.asdict()
        content['header']['date'] = content['header']['date'].strftime(DEFAULT_DATE_FORMAT)
        content['header']['gravity'] = list(content['header']['gravity'])

        # Constraints
        content['constraints'] = []
        joint_constraint = dict(type='Joint', parts=list())  # Multiple joint constraints can be merged in a single one
        for constraint in self.constraints:
            info = constraint.to_json_data()
            if info.get('type', None) == 'Joint':
                joint_constraint['parts'].extend(info['parts'])
            else:
                content['constraints'].append(info)
        if len(joint_constraint['parts']) > 0:
            content['constraints'].append(joint_constraint)

        # Obstacles
        content['obstacles'] = [obstacle.to_json_data() for obstacle in self.obstacles]

        # Objects
        # TODO: Implement Objects in Tasks

        # Goals
        content['goals'] = [goal.to_json_data() for goal in self.goals]
        return content

    @property
    def collision_objects(self) -> Tuple[hppfcl.CollisionObject, ...]:
        """All hppfcl collision objects by obstacles in the task"""
        warnings.warn("New feature collision objects - so far not tested!")
        return tuple(itertools.chain.from_iterable(obstacle.collision_objects for obstacle in self.obstacles))

    @property
    def goals_by_id(self) -> Dict[str, 'Goals.GoalBase']:
        """Return a mapping from all goal ID's to their python instances"""
        return {g.id: g for g in self.goals}

    @property
    def id(self) -> str:
        """The unique ID of the task"""
        return self.header.ID

    @property
    def name(self) -> str:
        """Returns the task ID, use not recommended"""
        warnings.warn("Tasks should be identified by ID.", category=DeprecationWarning)
        return self.id

    @property
    def obstacle_names(self) -> Tuple[str, ...]:
        """Returns the names of all obstacles in the task"""
        return tuple(o.name for o in self.obstacles)

    @property
    def base_constraint(self) -> Constraints.BasePlacement:
        """
        Returns the base constraint of this task.

        :raise: AttributeError if not BasePlacement Constraint given.
        """
        base = [x for x in self.constraints if isinstance(x, Constraints.BasePlacement)]
        if len(base) == 1:
            return base[0]
        else:
            raise AttributeError("Base constraint not unique within this task")

    def visualize(self,
                  viz: pin.visualize.MeshcatVisualizer = None,
                  robots: Optional[robot_or_robots] = None,
                  show_obstacle: bool = True,
                  show_goals: bool = True,
                  show_constraints: bool = True,
                  center_view: bool = True
                  ) -> pin.visualize.MeshcatVisualizer:
        """
        Plots the task in a Meshcat visualizer.

        Calls the "visualize" function for each Object, Robot, Goal and Constraint in the task.
        As Objects are the high-level representation of "things" in the Environment, this visualization function uses
        their high-level plotting functionalities, so visual meshes etc. will be chosen if available.

        :param viz: MeshcatViusalizer to work in; default: create new one
        :param robots: Robots to be visualized
        :param show_obstacle: bool to turn on obstacle rendering (default true)
        :param show_goals: bool to turn on goal rendering (default true)
        :param show_constraints: bool to turn on constraint rendering for those that can be displayed (default true)
        :param center_view: if set to true try to find  a reasonable camera position such that task visible, i.e.
          try to center camera on the first robot's base and if that is not set try to center the base constraint pose.
        """
        if viz is None:
            viz = pin.visualize.MeshcatVisualizer()
            viz.initViewer()

        if show_obstacle:
            for obj in self.obstacles:
                obj.visualize(viz)

        if robots is not None:
            if isinstance(robots, RobotBase):
                robots.visualize(viz)
                if center_view:
                    center_camera(viz.viewer, robots.placement.translation)
            else:
                if center_view:
                    center_camera(viz.viewer, robots[0].placement.translation)
                for robot in robots:
                    robot.visualize(viz)

        if show_goals:
            for goal in self.goals:
                if type(goal) is Goals.ReturnTo:
                    continue  # Cannot be visualized in a task
                goal.visualize(viz, scale=0.2)

        if show_constraints:
            for constraint in self.constraints:
                try:
                    constraint.visualize(viz, scale=0.2)
                except NotImplementedError:
                    # Not every constraint can be visualized, and that's okay
                    pass

        if center_view and robots is None:
            try:
                base_constraint = self.base_constraint
                center_camera(viz.viewer, base_constraint.base_pose.nominal.translation)
            except AttributeError:
                logging.info("Cannot recenter visualizer view to base.")

        return viz

    def __deepcopy__(self, memodict={}):
        """Custom deepcopy for a scneario class. Experimental!"""
        cpy = self.__class__(header=deepcopy(self.header))
        cpy.obstacles = [deepcopy(o) for o in self.obstacles]
        cpy.goals = tuple(deepcopy(t) for t in self.goals)
        cpy.constraints = tuple(deepcopy(c) for c in self.constraints)
        return cpy

    def __eq__(self, other):
        """Tasks are equal if headers, goals, constraints, and obstacles are equal."""
        if not isinstance(other, Task):
            return NotImplemented
        return (self.header == other.header
                and self.obstacles == other.obstacles
                and self.goals == other.goals
                and self.constraints == other.constraints)

    def __getstate__(self):
        """Return objects which will be pickled and saved."""
        return self.to_json_data()

    def __hash__(self):
        """Hash is given by the unique ID."""
        return hash(self.id)

    def __setstate__(self, state):
        """Take object from parameter and use it to retrieve class state."""
        cpy = self.__class__.from_json_data(state)
        self.__dict__ = cpy.__dict__
