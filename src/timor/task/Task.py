#!/usr/bin/env python3
# Author: Jonathan Külz
# Date: 17.01.22
from copy import deepcopy
from dataclasses import dataclass, field
import datetime
import itertools
import json
from pathlib import Path
from typing import Collection, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import warnings

from hppfcl import hppfcl
import numpy as np
import pinocchio as pin

from timor import compress_json_vectors, DEFAULT_DATE_FORMAT
from timor.Robot import PinRobot, RobotBase
from timor.task import Constraints, Goals
from timor.task.Obstacle import Obstacle
from timor.utilities.dtypes import TypedHeader, map2path
import timor.utilities.logging as logging

GRAVITY = np.array([0, 0, -9.81], dtype=float)  # Default gravity for z upwards TODO : Move to global configuration?!
robot_or_robots = Union[PinRobot, Iterable[PinRobot]]


@dataclass
class TaskHeader(TypedHeader):
    """The header every task contains"""

    ID: str
    version: str = 'Py'
    taskName: str = ""
    tags: List = field(default_factory=list)
    date: datetime.datetime = datetime.datetime(1970, 1, 1)
    timeStepSize: float = .01  # 10ms
    gravity: np.ndarray = field(default_factory=GRAVITY.copy)
    author: List[str] = TypedHeader.string_list_factory()
    email: List[str] = TypedHeader.string_list_factory()
    affiliation: List[str] = TypedHeader.string_list_factory()


class Task:
    """A task describes a problem that must be solved by one or multiple robots.

    The problem can be composed of multiple goals, each with their own constraints.
    Global constraints, e.g. valid base placements or the order of goals can be defined on the task-level.
    """

    _package_dir: Optional[Path]  # If a task was loaded from disk, this variable stored the path to package directory

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

        self._package_dir: Optional[Path] = None

    def __deepcopy__(self, memodict={}):
        """Custom deepcopy for a scneario class. Experimental!"""
        cpy = self.__class__(header=deepcopy(self.header))
        cpy.obstacles = [deepcopy(o) for o in self.obstacles]
        cpy.goals = [deepcopy(t) for t in self.goals]
        return cpy

    @classmethod
    def from_json_data(cls, data: Dict[str, any], package_dir: Union[Path, str]):
        """
        Loads a task from a parsed json (=dictionary).

        :param data: The parsed json data
        :param package_dir: The path to the package directory of contained mesh files.
        """
        content = deepcopy(data)  # Make sure we don't modify the original data while popping items
        header = TaskHeader(**{key: arg for key, arg in content.pop('header').items()})
        obstacles = [Obstacle.from_json_data(
            {**specs, **{'package_dir': package_dir}}) for specs in content.pop('obstacles')]
        goals = [Goals.GoalBase.goal_from_json_data(goal) for goal in content.pop('goals', [])]
        constraints = list()
        if 'Constraints' in content:
            logging.error("Constraints is written in upper case but should be lower case!")
            content['constraints'] = content.pop('Constraints')
        for desc in content.pop('constraints'):
            c = Constraints.ConstraintBase.from_json_data(desc)
            if type(c) is Constraints.AlwaysTrueConstraint:
                continue  # Just a placeholder for "basically no constraint"
            elif isinstance(c, Constraints.ConstraintBase):
                constraints.append(c)
            else:  # Some descriptions can contain multiple constraints
                constraints.extend([con for con in c if not isinstance(con, Constraints.AlwaysTrueConstraint)])
        if len(content) > 0:
            raise ValueError("Unresolved keys: {}".format(', '.join(content)))
        task = cls(header, obstacles, goals=goals, constraints=constraints)
        task._package_dir = package_dir
        return task

    @classmethod
    def from_json_file(cls, filepath: Union[Path, str], package_dir: Union[Path, str]):
        """
        Loads a task from a json file.

        :param filepath: The path to the json file
        :param package_dir: The path to the package directory, to which the mesh file paths are relative to.
        """
        filepath, package_dir = map(map2path, (filepath, package_dir))
        content = json.load(filepath.open('r'))
        return cls.from_json_data(content, package_dir)

    @staticmethod
    def from_srf_file(filepath: Path):
        """Future Work"""
        raise NotImplementedError()  # TODO

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

    def to_json_file(self, filepath: Path):
        """
        Write a task to a json file.

        :param filepath: The path to the file to write the json to
        """
        filepath = map2path(filepath)
        content = compress_json_vectors(json.dumps(self.to_json_data(), indent=2))
        if (self._package_dir is not None) and (not str(filepath).startswith(str(self._package_dir))):
            logging.info("Writing task to file outside of the package directory it was loaded from.")
        with filepath.open('w') as f:
            f.write(content)

    def to_json_string(self) -> str:
        """Returns the task as a json string"""
        return json.dumps(self.to_json_data())

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
    def base_constraint(self) -> Union['Constraints.BasePlacement', 'Constraints.AlwaysTrueConstraint']:
        """Returns the base constraint of this task"""
        base = [x for x in self.constraints if isinstance(x, Constraints.BasePlacement)]
        if len(base) == 1:
            return base[0]
        elif len(base) == 0:
            logging.info("Task has no base constraint; return always true constraint")
            return Constraints.AlwaysTrueConstraint()
        else:
            raise ValueError("Base constraint not unique within this task")

    def visualize(self,
                  viz: pin.visualize.MeshcatVisualizer = None,
                  robots: Optional[robot_or_robots] = None,
                  show_obstacle: bool = True,
                  show_goals: bool = True,
                  show_constraints: bool = True
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
            else:
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

        return viz
