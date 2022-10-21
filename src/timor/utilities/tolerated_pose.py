from __future__ import annotations

from typing import Dict, List, Optional, Union

from pinocchio.visualize import MeshcatVisualizer

from timor.task import Tolerance
from timor.utilities import logging
from timor.utilities.transformation import Transformation, TransformationLike


class ToleratedPose:
    """Defines a desired nominal placement/pose in world coordinates with a tolerance around."""

    nominal: Transformation  # The desired/nominal placement

    def __init__(self,
                 nominal: TransformationLike,
                 tolerance: Tolerance.ToleranceBase = None
                 ):
        """A tolerated placement defines a nominal placement and a volume around it which is considered valid.

        :param nominal: The nominal placement (placement-like, so a 4x4 transform).
        :param tolerance: The tolerance around the nominal placement. Can be any cartesian or rotation tolerance.
          Defaults to a narrow spatial tolerance.
        """
        self.nominal = Transformation(nominal)
        if tolerance is None:
            tolerance = Tolerance.DEFAULT_SPATIAL
        self.__set_tolerance(tolerance)

    def valid(self, other: Transformation) -> bool:
        """
        Returns whether the placement 'other' is within the tolerance of self.

        :param other: The placement to compare to.
        :return: True if the other is valid with regard to self, False otherwise.
        """
        return self.tolerance.valid(self.nominal, other)

    @classmethod
    def from_json_data(cls, description: Dict[str, any]) -> ToleratedPose:
        """Create a ToleratedPose from a json description.

        :param description: A json description as defined in the task documentation.
        :return: A ToleratedPose.
        """
        nominal = description['nominal']
        projections = description.get('toleranceProjection', ())
        tolerance_values = description.get('tolerance', ())
        if not len(projections) == len(tolerance_values):
            raise ValueError("The number of tolerance projections must match "
                             "the number of given tolerances (lower+upper).")
        tolerance = Tolerance.Composed([Tolerance.ToleranceBase.from_projection(p, v)
                                        for p, v in zip(projections, tolerance_values)])
        return cls(nominal, tolerance)

    @property
    def serialized(self) -> Dict[str, Union[List, str]]:
        """The json-compatible serialization of a placement with tolerance"""
        return {'nominal': self.nominal.serialized, **self.tolerance.to_projection()}

    @property
    def tolerance(self) -> Tolerance.ToleranceBase:
        """The tolerances of the placement."""
        return self._tolerance

    def visualize(self, viz: MeshcatVisualizer, name: str,
                  scale: float = 1., text: Optional[str] = None):
        """
        Draws this placement inside the visualizer object

        For detailed parameters refer to Transformation::visualize
        """
        self.nominal.visualize(viz, name, scale, text)
        logging.warning("Tolerated placement does not visualize tolerance yet")

    def __set_tolerance(self, value: Tolerance.ToleranceBase):
        """Type checking before setting the tolerance."""
        if isinstance(value, Tolerance.Composed):
            for t in value.tolerances:
                if not isinstance(t, (Tolerance.Cartesian, Tolerance.Rotation)):
                    raise ValueError(f'Transformation tolerance of type {type(t)} is not supported.')
        self._tolerance = value

    def __eq__(self, other):
        """Compares tolerated poses"""
        if type(other) != type(self):
            return NotImplemented
        return self.nominal == other.nominal and self.tolerance == other.tolerance

    def __getitem__(self, item):
        """Indexing a ToleratedPose defaults to indexing the nominal placement"""
        return self.nominal[item]
