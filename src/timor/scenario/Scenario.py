#!/usr/bin/env python3
# Author: Jonathan Külz
# Date: 17.01.22
from copy import deepcopy
import datetime
import itertools
import json
from pathlib import Path
from typing import Collection, Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple, Union
import warnings

from hppfcl import hppfcl
import numpy as np
import pinocchio as pin

import timor.utilities.logging as logging
from timor.Robot import PinRobot, RobotBase
from timor.scenario import Constraints, Goals
from timor.scenario.Obstacle import Obstacle

GRAVITY = np.array([0, 0, -9.81], dtype=float)  # Default gravity for z upwards TODO : Move to global configuration?!
robot_or_robots = Union[PinRobot, Iterable[PinRobot]]


class ScenarioHeader(NamedTuple):
    """The header every scenario contains"""

    scenarioID: str
    version: str = 'Py'
    tags: List = []
    date: datetime.datetime = datetime.datetime(1970, 1, 1)
    timeStepSize: float = .01  # 10ms
    gravity: np.ndarray = GRAVITY
    author: str = ''
    authorEMail: str = ''
    affiliation: str = ''


class Scenario:
    """A scenario describes a problem that must be solved by one or multiple robots.

    The problem can be composed of multiple goals, each with their own constraints.
    Global constraints, e.g. valid base placements or the order of goals can be defined on the scenario-level.
    For more information, refer to the crok documentation.
    """

    def __init__(self,
                 header: Union[ScenarioHeader, Dict],
                 obstacles: Collection[Obstacle] = None,
                 goals: Sequence['Goals.GoalBase'] = None,
                 constraints: Collection['Constraints.ConstraintBase'] = None
                 ):
        """Create a Scenario:

        :param header: The header contains meta informaton about the scenario. Must at least contain a scenarioID
        :param obstacles: Obstacles in the scenario/environment the robots should not collide with.
        :param goals: The complete set of goals that need to be achieved in order to solve the scenario
        :param constraints: "Global" constraint that must hold for the whole scenario.
        """
        self.header = header if isinstance(header, ScenarioHeader) else ScenarioHeader(**header)

        self.obstacles: List[Obstacle] = list()  # High-Level Objects like robots, obstacles
        self.goals: Tuple['Goals.GoalBase'] = tuple() if goals is None else tuple(goals)
        self.constraints: Tuple['Constraints.ConstraintBase'] = tuple() if constraints is None \
            else tuple(constraints)

        if obstacles is not None:
            for obstacle in obstacles:
                self.add_obstacle(obstacle)

    def __deepcopy__(self, memodict={}):
        """Custom deepcopy for a scneario class. Experimental!

        Deep copy of a scenario. DOES CURRENTLY NOT SUPPORT ROBOTS - a deep copy of a robot would remove
        the links between the robot obstacles and the pinnochio interface.
        """
        cpy = self.__class__(header=deepcopy(self.header))
        cpy.obstacles = [deepcopy(o) for o in self.obstacles]
        cpy.goals = [deepcopy(t) for t in self.goals]
        return cpy

    @classmethod
    def from_json(cls, filepath: Path, package_dir: Path):
        """
        Loads a scenario from a json file in crok-format.

        :param filepath: The path to the json file
        :param package_dir: The path to the package directory, to which the mesh file paths are relative to.
        """
        content = json.load(filepath.open('r'))
        if 'Tasks' in content or 'tasks' in content:
            raise NotImplementedError("Using tasks is deprecated! Define goals directly in scenarios.")
        ignore = ('authorInfo', 'scenarioName')
        content['Header']['scenarioID'] = str(content['Header']['scenarioID'])
        header = ScenarioHeader(**{key: arg for key, arg in content.pop('Header').items() if key not in ignore})
        obstacles = [
            Obstacle.from_crok_description(
                ID=specs['ID'],
                collision=specs['collision'],
                visual=specs.get('visual', None),
                name=specs.get('name', None),
                package_dir=package_dir
            ) for specs in content.pop('Obstacles')]
        goals = [Goals.GoalBase.from_crok_description(goal) for goal in content.pop('goals', [])]
        constraints = list()
        if 'constraints' in content:
            logging.warning("Constraints is written in lower case but should be upper case!")
            content['Constraints'] = content.pop('constraints')
        for desc in content.pop('Constraints'):
            c = Constraints.ConstraintBase.from_crok_description(desc)
            if type(c) is Constraints.AlwaysTrueConstraint:
                continue  # Just a placeholder for "basically no constraint"
            elif isinstance(c, Constraints.ConstraintBase):
                constraints.append(c)
            else:  # Some descriptions can contain multiple constraints
                constraints.extend([con for con in c if not isinstance(con, Constraints.AlwaysTrueConstraint)])
        if len(content) > 0:
            raise ValueError("Unresolved keys: {}".format(', '.join(content)))
        return cls(header, obstacles, goals=goals, constraints=constraints)

    @staticmethod
    def from_srf(filepath: Path):
        """Future Work"""
        raise NotImplementedError()  # TODO

    def add_obstacle(self, obstacle: Obstacle):
        """
        Adds an obstacle to the existing scenario.

        :param obstacle: Any obstacle derived from the Obstacle class.
          This can contain a single geometry or compositions of such.
        """
        if obstacle.id in self.obstacle_names:
            raise ValueError("An obstacle with the ID {} already exists in the Scenario.".format(obstacle.id))
        self.obstacles.append(obstacle)

    def to_dict(self) -> Dict[str, any]:
        """Returns the serialized, crok-compatible scenario dictionary"""
        content = dict()

        # Header
        content['Header'] = self.header._asdict()

        # Constraints
        content['Constraints'] = []
        joint_constraint = dict(type='Joint', parts=list())  # Multiple joint constraints can be merged in a single one
        for constraint in self.constraints:
            info = constraint.to_dict()
            if info.get('type', None) == 'Joint':
                joint_constraint['parts'].extend(info['parts'])
            else:
                content['Constraints'].append(info)
        if len(joint_constraint['parts']) > 0:
            content['Constraints'].append(joint_constraint)

        # Obstacles
        content['Obstacles'] = [obstacle.to_crok() for obstacle in self.obstacles]
        # content['Obstacles'] += [obstacle.to_crok() for obstacle in self.obstacles if
        #                          (type(obstacle) is ComposedObstacle)]

        # Objects
        # TODO: Implement Objects in Scenarios

        # Goals
        content['goals'] = [goal.to_dict() for goal in self.goals]
        return content

    def to_json(self, filepath: Path):
        """Write a scenario to a crok json file.

        Keep in mind that in crok, there is no pre-defined robot in a scenario, so the information about robots added
        will be lost.

        :param filepath: The path to the file to write the json to
        """
        json.dump(self.to_dict(), filepath.open('w'))

    @property
    def collision_objects(self) -> Tuple[hppfcl.CollisionObject, ...]:
        """All hppfcl collision objects by obstacles in the scenario"""
        warnings.warn("New feature collision objects - so far not tested!")
        return tuple(itertools.chain.from_iterable(obstacle.collision_objects for obstacle in self.obstacles))

    @property
    def goals_by_id(self) -> Dict[str, 'Goals.GoalBase']:
        """Return a mapping from all goal ID's to their python instances"""
        return {g.id: g for g in self.goals}

    @property
    def id(self) -> str:
        """The unique ID of the scenario"""
        return self.header.scenarioID

    @property
    def name(self) -> str:
        """Returns the scenario ID, use not recommended"""
        warnings.warn("Scenarios should be identified by ID.", category=DeprecationWarning)
        return self.id

    @property
    def obstacle_names(self) -> Tuple[str, ...]:
        """Returns the names of all obstacles in the scenario"""
        return tuple(o.name for o in self.obstacles)

    @property
    def base_constraint(self) -> Union['Constraints.BasePlacement', 'Constraints.AlwaysTrueConstraint']:
        """Returns the base constraint of this scenario"""
        base = [x for x in self.constraints if isinstance(x, Constraints.BasePlacement)]
        if len(base) == 1:
            return base[0]
        elif len(base) == 0:
            logging.info("Scenario has no base constraint; return always true constraint")
            return Constraints.AlwaysTrueConstraint()
        else:
            raise ValueError("Base constraint not unique within this scenario")

    def visualize(self,
                  viz: pin.visualize.MeshcatVisualizer = None,
                  robots: Optional[robot_or_robots] = None,
                  show_obstacle: bool = True,
                  show_goals: bool = True,
                  show_constraints: bool = True) -> pin.visualize.MeshcatVisualizer:
        """Plots the scenario in a Meshcat visualizer.

        Calls the "visualize" function for each Object, Robot, Goal and Constraint in the scenario.
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
                    continue  # Cannot be visualized in a scenario
                goal.visualize(viz, scale=0.2)

        if show_constraints:
            for constraint in self.constraints:
                try:
                    constraint.visualize(viz, scale=0.2)
                except NotImplementedError:
                    # Not every constraint can be visualized, and that's okay
                    pass

        return viz
