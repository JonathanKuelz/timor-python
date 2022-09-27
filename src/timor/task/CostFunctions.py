import abc

import numpy as np

from timor.task import Solution


class CostFunctionBase(abc.ABC):
    """
    A base class for cost functions that are evaluated on task solutions.

    By default, cost functions return positive values - if they describe something positive (e.g. number of goals
    fulfilled), the weight must be chosen negative.
    """

    default_weight = 1.

    def __init__(self, weight: float = default_weight):
        """create a weighted cost

        The default weight should be chosen in a way that higher cost means worse outcome.
        In case of rewards (e.g. cost proportional to goals achieved), the weight should default to a negative value.
        :param weight: Can be any number, positive or negative.
        """
        self.weight = weight

    """
    Allow basic maths with cost functions, the following should be possible:
    C_new = C_1 + C_2
    C_new = C_1 - C_2
    C_new = C_1 * 2.5
    C_new = C_1 / 2.5
    C_New = -C_1
    """

    def __add__(self, other):
        """The sum of two costs"""
        if isinstance(other, CostFunctionBase):
            return ComposedCost(self, other)
        return NotImplemented

    def __truediv__(self, other):
        """Change the cost function's weight by dividing it by a float"""
        if isinstance(other, (float, int)):
            return self.__class__(weight=self.weight / float(other))
        return NotImplemented

    def __mul__(self, other):
        """Change the cost function's weight by multiplying it by a float"""
        if isinstance(other, (float, int)):
            return self.__class__(weight=self.weight * float(other))
        return NotImplemented

    def __neg__(self):
        """Negate the cost function's weight (e.g. for rewards)"""
        return self.__class__(weight=-self.weight)

    def __sub__(self, other):
        """The difference of two costs"""
        if isinstance(other, CostFunctionBase):
            return ComposedCost(self, -other)

    @abc.abstractmethod
    def _evaluate(self, solution: 'Solution.SolutionBase') -> float:
        """Interface to define class-level evaluate functions"""
        pass

    def evaluate(self, solution: 'Solution.SolutionBase') -> float:
        """Calculates the cost of the solution for its internally stored task"""
        return self.weight * self._evaluate(solution)

    @staticmethod
    def from_descriptor(descriptor: str) -> 'CostFunctionBase':
        """Build the cost from a string:

        Takes a description string for a cost function and returns the corresponding instance
        :param descriptor: Abbreviation for the cost function. Format <name> or <name_weight>
        :return: instance of the cost function
        """
        abbreviations = {"cyc": CycleTime,
                         "mechEnergy": MechanicalEnergy,
                         "qDist": QDist,
                         "effort": Effort}
        references = descriptor.split(',')
        costs = list()
        for ref in references:
            class_specifier = ref.split("_")[0]
            try:
                class_ref = abbreviations[class_specifier]
            except KeyError:
                raise KeyError("Cost function not found: " + class_specifier)
            try:
                weight = float(ref.split("_")[1])
            except IndexError:
                weight = class_ref.default_weight
            costs.append(class_ref(weight=weight))
        if len(costs) == 1:
            return costs[0]
        return ComposedCost(*costs)


class ComposedCost(CostFunctionBase):
    """A cost function composed of different atomic cost functions"""

    default_weight = 1.

    def __init__(self, *args, weight: float = default_weight):
        """
        A composed cost is a container with multiple cost functions, providing a unified interface.
        """
        super().__init__(weight=weight)
        self._internal = list()
        for cost in args:
            if isinstance(cost, ComposedCost):
                self._internal.extend(cost._internal)
            else:
                if not isinstance(cost, CostFunctionBase):
                    raise TypeError("Unexpected input argument: {}".format(cost))
                self._internal.append(cost)

    def __truediv__(self, other):
        """Divide all internal cost functions by the given number."""
        if isinstance(other, (float, int)):
            return self.__class__(*self._internal, weight=self.weight / float(other))
        return NotImplemented

    def __mul__(self, other):
        """Multiply all internal cost functions by the given number."""
        if isinstance(other, (float, int)):
            return self.__class__(*self._internal, weight=self.weight * float(other))
        return NotImplemented

    def __neg__(self):
        """Negate all internal cost functions."""
        return self.__class__(*self._internal, weight=-self.weight)

    def _evaluate(self, solution: 'Solution.SolutionBase') -> float:
        """The evaluation of the composed cost function is the sum of all internal cost functions"""
        return sum(cost.evaluate(solution) for cost in self._internal)


class CycleTime(CostFunctionBase):
    """Increases with trajectory length needed to solve a task"""

    default_weight = 1.

    def _evaluate(self, solution: 'Solution.SolutionBase') -> float:
        """The total time needed by the solution"""
        return solution.trajectory.t[-1]


class Effort(CostFunctionBase):
    """Increases with the torques used by a solution"""

    default_weight = 1.

    def _evaluate(self, solution: 'Solution.SolutionBase') -> float:
        """The integral of effort over time in the solution"""
        return np.sum(np.abs(solution.torques)) * (solution.time_steps[1] - solution.time_steps[0])
        # TODO : No field for sample time? -or- move scale by time into sum


class GoalsFulfilled(CostFunctionBase):
    """Cost of alpha for every goal fulfilled in the task. (negative weight --> reward)"""

    default_weight = -1.

    def _evaluate(self, solution: 'Solution.SolutionBase') -> float:
        """The number of goals fulfilled"""
        reward = 0
        for goal in solution.task.goals:
            if goal.achieved(solution):  # Always returns 0 if the solution is not valid
                reward += 1
        return reward


class MechanicalEnergy(CostFunctionBase):
    """Increases with the mechanical energy used by a solution"""

    default_weight = 1.

    def _evaluate(self, solution: 'Solution.SolutionBase') -> float:
        """The integral of mechanical energy over time in the solution"""
        return np.sum(np.abs(np.einsum("ij,ij->i", solution.torques, solution.trajectory.dq))) * \
            (solution.time_steps[1] - solution.time_steps[0])
        # TODO: Implement for solutions where no trajectory is given, but the torque input is
        #  -> (solution.trajecory not available);
        #  suggestion: solution should integrate to get q, dq, ddq on demand as it does for torques already
        # TODO : No field for sample time? -or- move scale by time into sum


class NumJoints(CostFunctionBase):
    """Punishes a higher number of actuated joints"""

    default_weight = .1

    def _evaluate(self, solution: 'Solution.SolutionBase') -> float:
        """The number of actuated joints in the solution robot"""
        return solution.robot.njoints


class RobotMass(CostFunctionBase):
    """Punishes higher weight of the robot"""

    default_weight = .05

    def _evaluate(self, solution: 'Solution.SolutionBase') -> float:
        """The total mass of the solution robot"""
        return solution.robot.mass


class QDist(CostFunctionBase):
    """Increases with the joint space distance traversed by the solution"""

    default_weight = 1.

    def _evaluate(self, solution: 'Solution.SolutionBase') -> float:
        """The integral over time for all joint parameters (mixes up radian and meters!)"""
        return float(np.sum(np.abs(solution.trajectory.q[1:, :] -  # All but first q
                                   solution.trajectory.q[:-1, :])))  # All but last q
        # TODO : No field for sample time? -or- move scale by time into sum
