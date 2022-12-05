#!/usr/bin/env python3
# Author: Jonathan Külz
# Date: 03.03.22
from typing import Union

import meshcat.animation
import meshcat.geometry
import numpy as np
from pinocchio import COLLISION, VISUAL  # noqa: F401
import pinocchio as pin
from pinocchio.visualize.meshcat_visualizer import MeshcatVisualizer, isMesh

from timor.utilities import spatial, transformation


class MeshcatVisualizerWithAnimation(MeshcatVisualizer):
    """
    Extends Pinocchio's MeshcatVisualizer by create_animation_frame.

    This creates an animation from the current state of all pinocchio robots within the scene;
    """

    @classmethod
    def from_MeshcatVisualizer(cls, vis: MeshcatVisualizer) -> 'MeshcatVisualizerWithAnimation':
        """Turn a normal MeshcatVisualizer into a MeshcatVisualizerWithAnimation"""
        if isinstance(vis, MeshcatVisualizerWithAnimation):
            return vis
        new_vis = cls()
        # Copy all values of A to B
        # It should not have any problem since they have common template
        for key, value in vis.__dict__.items():
            new_vis.__dict__[key] = value
        return new_vis

    def create_animation_frame(self, geometry_type, frame):
        """
        Can be used to capture the current visualizer state as an animation frame
        """
        if geometry_type == pin.GeometryType.VISUAL:
            geom_model = self.visual_model
            geom_data = self.visual_data
        else:
            geom_model = self.collision_model
            geom_data = self.collision_data

        pin.updateGeometryPlacements(self.model, self.data, geom_model, geom_data)
        for visual in geom_model.geometryObjects:
            visual_name = self.getViewerNodeName(visual, geometry_type)
            # Get mesh placement.
            M = geom_data.oMg[geom_model.getGeometryId(visual.name)]
            # Manage scaling: force scaling even if this should be normally handled by MeshCat (but there is a bug here)
            if isMesh(visual):
                scale = np.asarray(visual.meshScale).flatten()
                S = np.diag(np.concatenate((scale, [1.0])))
                T = np.array(M.homogeneous).dot(S)
            else:
                T = M.homogeneous

            # Update viewer configuration.
            frame[visual_name].set_transform(T)


def place_arrow(
        viz: MeshcatVisualizer,
        name: str,
        material: meshcat.geometry.Material = meshcat.geometry.MeshBasicMaterial(),
        scale: float = 1.,
        placement: transformation.TransformationLike = transformation.Transformation.neutral(),
        axis: str = 'z') -> None:
    """
    Creates a composed meshcat geometry object that looks like an arrow.

    :param viz: Visualizer instance to draw into
    :param name: Unique object name for the visualizer
    :param material: If provided, this defines the appearance (e.g. color)
    :param scale: Arrow length in meters will be 0.1 * scale
    :param placement: Defines orientation and placement of the arrow.
    :param axis: Axis alignment of the arrow. The default is alignment to the z-axis in the "placement" coordinate frame
    """
    placement = transformation.Transformation(placement)
    if axis == 'x':
        rot = spatial.rotZ(-np.pi / 2)
    elif axis == 'y':
        rot = transformation.Transformation.neutral()
    elif axis == 'z':
        rot = spatial.rotX(np.pi / 2)
    else:
        raise ValueError("Invalid argument for axis: {}".format(axis))

    length = .1 * scale
    base_length = length * 3 / 5
    base_width = length / 7
    head_length = length * 2 / 5
    rmax_head = 1.5 * length / 5
    base = meshcat.geometry.Cylinder(base_length, base_width)
    head = meshcat.geometry.Cylinder(head_length, radiusTop=0, radiusBottom=rmax_head)

    viz.viewer[name + '_arr_body'].set_object(base, material)
    base_transform = placement @ rot @ spatial.homogeneous(translation=[0, -.5 * base_length - head_length, 0])
    viz.viewer[name + '_arr_body'].set_transform(base_transform)

    viz.viewer[name + '_arr_head'].set_object(head, material)
    head_transform = placement @ rot @ spatial.homogeneous(translation=[0, -.5 * head_length, 0])
    viz.viewer[name + '_arr_head'].set_transform(head_transform)


def drawable_coordinate_system(placement: transformation.TransformationLike,
                               scale: float = 1.) -> meshcat.geometry:
    """
    A visual representation of the origin of a coordinate system.

    The coordinate axis are drawn as three lines in red, green, and blue along the x, y, and z axes. The `scale`
    parameter controls the length of the three lines.
    Returns an `Object` which can be passed to `set_object()`
    Other than meshcat.geometry.triad, this allows drawing the triad in any coordinate system, which is
    defined by <original coordinate system @ transformation>.

    :param placement: 4x4 homogeneous transformation
    :param scale: Length of the drawn vectors for the coordinate system main axes
    :return: A meshcat object that can be visualized via viewer[your_name].set_object(this)
    """
    placement = transformation.Transformation(placement)
    p0 = (placement @ transformation.Transformation.neutral())[:3, 3]
    x = (placement @ spatial.homogeneous(translation=[scale, 0, 0]))[:3, 3]
    y = (placement @ spatial.homogeneous(translation=[0, scale, 0]))[:3, 3]
    z = (placement @ spatial.homogeneous(translation=[0, 0, scale]))[:3, 3]
    return meshcat.geometry.LineSegments(
        geometry=meshcat.geometry.PointsGeometry(
            position=np.array([p0, x, p0, y, p0, z], dtype=np.float32).T,
            color=np.array([
                [1, 0, 0], [1, 0.6, 0],
                [0, 1, 0], [0.6, 1, 0],
                [0, 0, 1], [0, 0.6, 1]]).astype(np.float32).T
        ),
        material=meshcat.geometry.LineBasicMaterial(vertexColors=True)
    )


def animation(robot: 'Robot.PinRobot', q: np.ndarray, dt: float,  # pragma: no cover # noqa: F821
              visualizer: Union[pin.visualize.MeshcatVisualizer,
                                MeshcatVisualizerWithAnimation] = None) -> MeshcatVisualizerWithAnimation:
    """
    Creates an animation of a robot movement.

    :param robot: The robot to animate
    :param q: The joint angles to animate
    :param dt: The time step between frames
    :param visualizer: If given, the movie will be generated in the existing visualizer
    """
    if q.shape[1] != robot.njoints:
        raise ValueError("The provided configurations must be of shape time steps x dof")

    robot.update_configuration(q[0, :])
    viz = MeshcatVisualizerWithAnimation.from_MeshcatVisualizer(robot.visualize(visualizer))
    anim = meshcat.animation.Animation(default_framerate=1 / dt)
    for i in range(0, q.shape[0]):
        with anim.at_frame(viz.viewer, i) as frame:
            robot.update_configuration(q[i, :])
            viz.create_animation_frame(pin.VISUAL, frame)

    viz.viewer.set_animation(anim)
    return viz


class TextTexture(meshcat.geometry.Texture):
    """
    Forwarded from meshcat-py master branch

    https://github.com/rdeits/meshcat-python/blob/master/src/meshcat/geometry.py
    """

    def __init__(self, text, font_size=100, font_face='sans-serif'):
        """Create texture with text rendered."""
        super(TextTexture, self).__init__()
        self.text = text
        # font_size will be passed to the JS side as is; however if the
        # text width exceeds canvas width, font_size will be reduced.
        self.font_size = font_size
        self.font_face = font_face

    def lower(self, object_data):
        """Internal function used to create dict for underlying visualizer"""
        return {
            u"uuid": self.uuid,
            u"type": u"_text",
            u"text": self.text,
            u"font_size": self.font_size,
            u"font_face": self.font_face,
        }


def scene_text(text: str, size: float = 1., **kwargs):
    """
    Create a cube geometry with text displayed on sides and size edge length.

    Altered from meshcat-py master branch
    https://github.com/rdeits/meshcat-python/blob/master/src/meshcat/geometry.py

    :todo: Only have one rightly oriented TextTexture...
    """
    return meshcat.geometry.Mesh(meshcat.geometry.Box((size, 0, size)),
                                 meshcat.geometry.MeshPhongMaterial(map=TextTexture(text, **kwargs),
                                                                    transparent=True,
                                                                    needsUpdate=True))
