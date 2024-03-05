#!/usr/bin/env python3
# Author: Matthias Mayer, Jonathan Külz
# Date: 21.07.23
from __future__ import annotations

from copy import deepcopy
import itertools
from typing import Dict, Iterator, Optional, Union
import uuid

import meshcat
import numpy as np
from pinocchio.visualize import MeshcatVisualizer

from timor.utilities.transformation import Transformation, TransformationLike
from timor.utilities.jsonable import JSONable_mixin
from timor.utilities.dtypes import SingletonMeta

FRAME_ID_TYPE = str


class Frame(JSONable_mixin):
    """
    A frame is a position and orientation in a 3D world. It is defined by a transformation relative to a parent frame.

    All frames span up a tree with a single root node: the world frame.
    The following special frames exist:

      - `world`: denotes the world frame.
      - `nominal`: denotes the nominal placement of a tolerated pose.
    """

    def __init__(self, ID: FRAME_ID_TYPE, transformation: TransformationLike, parent: Frame):
        """
        Create a frame with a given transformation relative to its parent frame.

        :param ID: The ID of the frame as a tuple of strings. IDs with . are special for auto-generation
            (see frame-tree).
        :param transformation: The transformation from the parent frame to this frame.
        :param parent: The parent frame.
        """
        if ID in SPECIAL_FRAMES:
            raise ValueError(f"The frame ID {ID} is reserved for the corresponding Singleton special frame.")
        self._transformation: Transformation = Transformation(transformation)
        self._parent: Frame = parent
        self._ID: FRAME_ID_TYPE = ID

    @property
    def absolute(self) -> Transformation:
        """Return the transformation from the world frame to this frame."""
        return self.in_world_coordinates()

    @property
    def ID(self) -> FRAME_ID_TYPE:
        """Return the name of the frame."""
        return self._ID

    @property
    def parent(self) -> Frame:
        """Return the parent frame."""
        return self._parent

    @property
    def parent2frame(self) -> Transformation:
        """Return the transformation from the parent frame to this frame."""
        return self._transformation

    @property
    def frame2parent(self) -> Transformation:
        """Return the transformation from this frame to the parent frame."""
        return self.parent2frame.inv

    def in_world_coordinates(self) -> Transformation:
        """
        Return the transformation from the world frame to this frame.
        """
        if self is WORLD_FRAME:
            return Transformation.neutral()
        return self.parent.absolute @ self.parent2frame

    def rotate_to_this_frame(self, t: Transformation, old_basis: Union[Frame, Transformation]) -> Transformation:
        r"""
        This method performs a change of basis using a similarity transformation.

        Given a transformation t, expressed in old_basis, this method assumes a new basis frame, centered in the same
        point as old_basis, but with axis aligned to the ones of this frame. This is especially useful when there's a
        need to express deviations between a desired and an actual pose (with delta t) in an unrelated (e.g., WORLD)
        frame.
        :param t: The transformation to be transformed.
        :param old_basis: The frame (or the world2frame transformation for an imaginary frame) t is expressed in.
        :return: t in a new basis with axis aligned to this frame.
        """
        if isinstance(old_basis, Frame):
            old_basis = old_basis.absolute
        # For t* expressed in F1, R=rot(F1, F2), t = R' t* R is expressed in the rotated frame F2
        # Here, we are reverting this basis transformation to get t*
        this_orientation = Transformation.from_rotation(self.absolute.rotation)
        return this_orientation.inv @ WORLD_FRAME.rotate_to_this_frame(t, old_basis) @ this_orientation

    def to_json_data(self):
        """Return a json representation of the frame as used by a (tolerated) pose, i.e., without a name / ID."""
        ret = {"transformation": self.parent2frame.to_json_data()}
        if self.parent is not WORLD_FRAME:
            ret["parent"] = self.parent.ID
        return ret

    def visualize(self, viz: MeshcatVisualizer, name_space: str = None, *args, **kwargs):
        """
        Visualize the frame like a transformation.

        :param viz: The visualizer to use.
        :param name_space: The prefix in viz to use for the frame.
        :param kwargs: Additional arguments passed to visualize of underlying transformation.
        """
        name = f"{name_space}/{self.ID}" if name_space is not None else self.ID
        self.absolute.visualize(viz, name, *args, **kwargs)

    def __eq__(self, other):
        """Two frames are equal if they represent the same reference coordinate system."""
        if type(self) is not type(other):
            return NotImplemented
        try:
            return self.absolute == other.absolute
        except RecursionError:
            id1, id2 = self.ID, other.ID
            raise ValueError(f"The two frames compared, {id1} and {id2}, are in a loop (not a proper frame tree).")

    def __hash__(self):
        """Return a hash of the frame."""
        return hash(self.ID)

    def __matmul__(self, other: Transformation) -> Transformation:
        """Returns the absolute transformation in world coordinates, assuming `other` is given in this frame."""
        if isinstance(other, Transformation):
            return self.absolute @ other
        return NotImplemented

    def __repr__(self):
        """Return a debug representation of the frame."""
        return f"Frame {self.ID}: ({self.parent2frame}, {self.parent})"

    def __str__(self):
        """Return a string representation of the frame by giving its absolute pose."""
        if self.ID in SPECIAL_FRAMES:
            return f"Special frame {self.ID}"
        return self.absolute.__str__()


class NominalFrameType(Frame, metaclass=SingletonMeta):
    """
    The nominal frame is a special frame that is used to represent the nominal placement of a tolerated pose.

    Opposed to any other frame, it is not fixed in space: It transforms with the pose it represents. It is implemented
    as a singleton, i.e., there is only one NominalFrame, but it can represent any number of poses.
    """

    UndefinedError = AttributeError("The nominal frame does not have a well-defined placement.")

    def __init__(self):
        """We will create the nominal frame exactly once."""
        object.__init__(self)  # We don't want to use the default Frame init
        self._ID = 'nominal'

    @property
    def parent(self) -> Frame:
        """The nominal's frame parent is the world frame, always."""
        return WORLD_FRAME

    @property
    def parent2frame(self) -> Transformation:
        """This is not defined"""
        raise self.UndefinedError

    def in_world_coordinates(self) -> Transformation:
        """This method is not well-defined."""
        raise self.UndefinedError

    def rotate_to_this_frame(self, t: Transformation, old_basis: Transformation = None) -> Transformation:
        """For the nominal frame, no matter the old basis, we will just use it."""
        return t

    def to_json_data(self):
        """The nominal frame cannot be serialized, it is used by tolerances on-the-fly only."""
        raise AttributeError("A nominal frame cannot be serialized to JSON.")

    def visualize(self, viz: MeshcatVisualizer, name_space: str = None, *args, **kwargs):
        """A nominal frame doesn't have a fixed placement and can't be visualized."""
        raise AttributeError("A nominal frame doesn't have a fixed placement and can't be visualized.")

    def __eq__(self, other):
        """Singleton equality means same instance."""
        if not isinstance(other, Frame):
            return NotImplemented
        return self is other

    def __repr__(self):
        """There is not much debug information about the nominal frame."""
        return str(self)


class WorldFrameType(Frame, metaclass=SingletonMeta):
    """
    The world frame is a special frame that always represents the root node in a frame tree. It's the "world origin".
    """

    RootError = AttributeError("The world frame is the root node of any frame tree. It has no parent.")

    def __init__(self):
        """We will create the world frame exactly once"""
        object.__init__(self)  # We don't want to use the default Frame init
        self._ID = 'world'

    @property
    def parent(self) -> Frame:
        """The world frame has no parent"""
        raise self.RootError

    @property
    def parent2frame(self) -> Transformation:
        """This is not defined"""
        raise self.RootError

    def in_world_coordinates(self) -> Transformation:
        """Neutral"""
        return Transformation.neutral()

    def rotate_to_this_frame(self, t: Transformation, old_basis: Union[Frame, Transformation]) -> Transformation:
        """For the world frame, this method only needs to 'neutralize' the old_basis rotation"""
        if old_basis is self:
            return t
        elif isinstance(old_basis, Frame):
            old_basis = old_basis.absolute
        other_orientation = Transformation.from_rotation(old_basis.rotation)
        return other_orientation @ t @ other_orientation.inv

    def to_json_data(self):
        """The world frame is the only frame without a parent."""
        return {"transformation": Transformation.neutral().to_json_data()}

    def __repr__(self):
        """Return a debug representation of the frame."""
        return str(self)


class FrameTree(JSONable_mixin):
    """
    A tree of frames with a single root (the world frame).

    :note: . is a reserved keyword in frame IDs to have namespace for auto-generated frames such as goal.goal_ID frame.
        They are especially useful when serializing the frame tree as these can be omitted.
    """

    def __init__(self, frames: Dict[FRAME_ID_TYPE, Dict[str, TransformationLike]]):
        """
        Create a frame tree from a dictionary of frames.

        :param frames: A dictionary with key = frame name and value = dictionary with keys "parent" and "transformation"
          to create each frame from.
        """
        if "world" in frames:
            raise ValueError("The world frame is reserved and must not be provided externally.")
        if "nominal" in frames:
            raise ValueError("The nominal frame is reserved and must not be used.")
        if any('.' in ID for ID in frames.keys()):
            raise ValueError("Frame names must not contain a period.")

        self._known_frames: Dict[FRAME_ID_TYPE, Frame] = SPECIAL_FRAMES.copy()
        _frames = deepcopy(frames)
        for _ in range(len(frames)):  # this is the maximum number of possible iterations if everything is correct
            new_frames = set()
            for ID, frame in _frames.items():
                if frame["parent"] in self._known_frames:
                    self._known_frames[ID] = Frame(ID, frame.get("transformation", Transformation.neutral()),
                                                   self._known_frames[frame["parent"]])
                    new_frames.add(ID)
            for ID in new_frames:  # Remove all new frames from the input dictionary
                _frames.pop(ID)
            if len(_frames) == 0:
                break  # Base case
            if len(new_frames) == 0:
                raise ValueError(f"Invalid frame tree contains a closed cycle: {_frames}.")
        else:
            if len(frames) > 0:
                raise ValueError(f"Invalid frame tree contains a closed cycle: {_frames}.")

    @classmethod
    def empty(cls) -> FrameTree:
        """Create an empty frame tree."""
        return cls({})

    @classmethod
    def from_json_data(cls, d: Dict, *args, **kwargs) -> 'FrameTree':
        """Create a frame tree from a dictionary of frames."""
        return cls(d)

    def add(self,
            ID: Optional[FRAME_ID_TYPE],
            transformation: TransformationLike,
            parent: Union[FRAME_ID_TYPE, Frame]) -> Frame:
        """
        Create and add a new frame to the tree.

        :param ID: The ID of the frame. If None, a random ID is generated.
        :param transformation: The transformation from the parent frame to this frame.
        :param parent: The parent frame of the fresh generated frame. Can either be an ID or a frame instance. The
            parent must already be part of the tree.
        """
        if ID in self:
            raise ValueError(f"Frame {ID} already exists.")
        if ID is None:
            ID = f"unnamed.{uuid.uuid4()}"
        if isinstance(parent, Frame):
            parent_id = parent.ID
        else:
            parent_id = parent
        if parent_id not in self:
            raise ValueError(f"The parent frame {parent_id} is not part of this tree.")

        if isinstance(parent, Frame):
            if not self[parent_id] is parent:
                raise ValueError(f"A frame with ID {parent_id} exists in this tree, but it is not the frame that was "
                                 f"provided.")
        parent: Frame = self[parent]
        frame = Frame(ID, transformation, parent)
        self._known_frames[ID] = frame
        return frame

    def to_json_data(self) -> Dict[str, Dict[str, TransformationLike]]:
        """Return a dictionary of primary frames, i.e., those without a dot in their ID."""
        json_frames = {f for ID, f in self._known_frames.items() if ID not in SPECIAL_FRAMES and '.' not in ID}
        frames = {frame.ID: {"parent": frame.parent.ID, "transformation": frame.parent2frame.to_json_data()}
                  for frame in json_frames}
        return frames

    def visualize(self, viz: MeshcatVisualizer, scale: float = 1., name_space: str = "FrameTree",
                  line_color: Optional[np.ndarray] = None):  # pragma: no cover
        """
        Draws this frame tree inside the visualizer object

        :param viz: Visualizer to add frame tree to.
        :param scale: Size in meter of each axis of placement
        :param name_space: Name to add geometry of frame tree to the viewer with (overwrites existing!)
        :param line_color: Color of the line between frames as RGB from 0 to 1
        """
        for frame in self:
            frame.visualize(viz, scale=scale, name_space=name_space)

        # Add line between frames
        line_segments = itertools.chain.from_iterable(
            (frame.parent.in_world_coordinates().translation, frame.in_world_coordinates().translation)
            for frame in self if frame is not WORLD_FRAME)
        line_segments = np.fromiter(line_segments, dtype=np.dtype((np.float32, 3))).T

        if line_color is None:
            line_color = np.ones(line_segments.shape)
        else:
            line_color = np.tile(line_color, (line_segments.shape[1], 1)).T
        line = meshcat.geometry.LineSegments(
            geometry=meshcat.geometry.PointsGeometry(position=line_segments, color=line_color),
            material=meshcat.geometry.LineBasicMaterial(vertexColors=True)
        )
        viz.viewer[f"{name_space}/connectivity"].set_object(line)

    def __contains__(self, frame: Union[Frame, FRAME_ID_TYPE]) -> bool:
        """Returns whether the given frame is known."""
        if isinstance(frame, Frame):
            frame = frame.ID
        return frame in self._known_frames

    def __eq__(self, other) -> bool:
        """Returns whether the two frame trees are equal."""
        if type(self) is not type(other):
            return NotImplemented  # pragma: no cover
        return self._known_frames == other._known_frames

    def __getitem__(self, frame_name: FRAME_ID_TYPE) -> Frame:
        """Returns the given frame."""
        return self._known_frames[frame_name]

    def __iter__(self) -> Iterator[Frame]:
        """Returns an iterator over the frames in the tree."""
        return (v for k, v in self._known_frames.items() if k not in {"nominal"})

    def __len__(self) -> int:
        """Returns the number of frames in the tree."""
        return len(self._known_frames)


WORLD_FRAME = WorldFrameType()
NOMINAL_FRAME = NominalFrameType()
SPECIAL_FRAMES = {'world': WORLD_FRAME, 'nominal': NOMINAL_FRAME}
