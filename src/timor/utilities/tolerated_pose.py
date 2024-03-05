from __future__ import annotations

import itertools
import json
from typing import Dict, List, Union

from pinocchio.visualize import MeshcatVisualizer

from timor.task import Tolerance
from timor.utilities.frames import Frame, FrameTree, NOMINAL_FRAME, WORLD_FRAME
from timor.utilities.jsonable import JSONable_mixin
from timor.utilities.transformation import Transformation


class ToleratedPose(JSONable_mixin):
    """Defines a desired nominal placement/pose in world coordinates with a tolerance around."""

    nominal: Frame  # The desired/nominal placement

    def __init__(self,
                 nominal: Union[Transformation, Frame],
                 tolerance: Tolerance.ToleranceBase = None
                 ):
        """A tolerated placement defines a nominal placement and a volume around it which is considered valid.

        :param nominal: The nominal placement, either as a frame or a transformation assumed to be relative to the world
            frame.
        :param tolerance: The tolerance around the nominal placement. Can be any cartesian or rotation tolerance.
            Defaults to a narrow spatial tolerance.
        """
        if isinstance(nominal, Transformation):
            nominal = Frame("", nominal, parent=WORLD_FRAME)
        self.nominal = nominal
        if tolerance is None:
            tolerance = Tolerance.DEFAULT_SPATIAL
        self.__set_tolerance(tolerance)

    def valid(self, other: Transformation) -> bool:
        """
        Returns whether the placement 'other' is within the tolerance of self.

        :param other: The placement to compare to.
        :return: True if the other is valid with regard to self, False otherwise.
        """
        return self.tolerance.valid(self.nominal.in_world_coordinates(), other)

    @classmethod
    def from_json_data(cls, d: Dict[str, any], *args, **kwargs) -> ToleratedPose:
        """Create a ToleratedPose from a json description.

        :param d: A json description as defined in the task documentation.
        :param kwargs:
          * 'frameName': The name of the nominal frame.
          * 'frames': A FrameTree to resolve add / find the nominal frame.
          * 'annonymousFrame': If True, the nominal frame will not be added to the FrameTree.
        :return: A ToleratedPose.
        """
        nominal = d['nominal']
        nominal_parent = d.get('nominalParent', 'world')
        frame_name = kwargs.get('frameName', None)
        if 'frames' not in kwargs and nominal_parent != 'world':
            raise ValueError("If the nominal parent is not 'world', a FrameTree must be provided.")
        frames: FrameTree = kwargs.get('frames', FrameTree.empty())
        if kwargs.get('annonymousFrame', False):
            nominal = Frame("", nominal, parent=frames[nominal_parent])
        else:
            if frame_name in frames:
                nominal = frames[frame_name]
            else:
                nominal = frames.add(frame_name, nominal, nominal_parent)
        projections = d.get('toleranceProjection', ())
        tolerance_values = d.get('tolerance', ())
        tolerance_frames = d.get('toleranceFrame', tuple(itertools.repeat(NOMINAL_FRAME, len(projections))))
        tolerance_frames = tuple(map(lambda f: frames[f] if isinstance(f, str) else f, tolerance_frames))
        if len(projections) != len(tolerance_values):
            raise ValueError("The number of tolerance projections must match "
                             "the number of given tolerances (lower+upper).")
        if len(projections) != len(tolerance_frames):
            raise ValueError("The number of tolerance projections must match "
                             "the number of given frames.")
        tolerance = Tolerance.Composed([Tolerance.ToleranceBase.from_projection(p, v, f)
                                        for p, v, f in zip(projections, tolerance_values, tolerance_frames)])
        return cls(nominal, tolerance)

    def to_json_data(self) -> Dict[str, Union[List, str]]:
        """The json-compatible serialization of a placement with tolerance"""
        frame_data = self.nominal.to_json_data()
        # Repackage for nominal pose
        if 'parent' in frame_data:
            frame_data['nominalParent'] = frame_data.pop('parent')
        frame_data['nominal'] = frame_data.pop('transformation')
        return {**frame_data, **self.tolerance.to_projection()}

    @property
    def tolerance(self) -> Tolerance.ToleranceBase:
        """The tolerances of the placement."""
        return self._tolerance

    def to_json_string(self) -> str:
        """Returns the json string representation of this placement."""
        return json.dumps(self.to_json_data(), indent=2)

    def visualize(self, viz: MeshcatVisualizer, name: str, scale: float = 1., **kwargs):  # pragma: no cover
        """
        Draws this placement inside the visualizer object

        For detailed kwargs refer to Transformation::visualize
        """
        if isinstance(self.nominal, Frame):
            self.nominal.in_world_coordinates().visualize(viz, name, scale, **kwargs)
        else:
            self.nominal.visualize(viz, name, scale, **kwargs)
        self.tolerance.visualize(viz, self.nominal.in_world_coordinates().homogeneous, name=name + '_tol')

    def __set_tolerance(self, value: Tolerance.ToleranceBase):
        """Type checking before setting the tolerance."""
        if isinstance(value, Tolerance.Composed):
            for t in value.tolerances:
                if not isinstance(t, (Tolerance.Cartesian, Tolerance.Rotation)):
                    raise ValueError(f'Transformation tolerance of type {type(t)} is not supported.')
        self._tolerance = value

    def __eq__(self, other):
        """Compares tolerated poses"""
        if type(other) is not type(self):
            return NotImplemented
        return self.nominal == other.nominal and self.tolerance == other.tolerance

    def __getitem__(self, item):
        """Indexing a ToleratedPose defaults to indexing the nominal placement"""
        return self.nominal.in_world_coordinates()[item]

    def __str__(self):
        """Make human-readable."""
        return f"Tolerated pose @ {self.nominal} with tolerance {self._tolerance}"

    def __repr__(self):
        """Make human-readable in debugger, etc."""
        return str(self)
