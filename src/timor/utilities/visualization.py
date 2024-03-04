#!/usr/bin/env python3
# Author: Jonathan Külz
# Date: 03.03.22
from pathlib import Path
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import matplotlib
from matplotlib import pyplot as plt
import meshcat.animation
import meshcat.geometry
import numpy as np
from pinocchio import COLLISION, VISUAL  # noqa: F401
import pinocchio as pin
from pinocchio.visualize.meshcat_visualizer import MeshcatVisualizer, isMesh

from timor import ModuleAssembly
from timor.utilities import logging, module_classification, spatial, transformation
from timor.utilities.dtypes import timeout
from timor.utilities.file_locations import map2path
from timor.utilities.transformation import Transformation, TransformationConvertable

DEFAULT_COLOR_MAP = {  # Type enumeration from module_classification - import not possibility (circular import)
    module_classification.ModuleType.BASE: np.array([165 / 255, 165 / 255, 165 / 255, 1.]),
    module_classification.ModuleType.LINK: np.array([127 / 255, 127 / 255, 127 / 255, 1.]),
    module_classification.ModuleType.JOINT: np.array([248 / 255, 135 / 255, 1 / 255, 1.]),
    module_classification.ModuleType.END_EFFECTOR: np.array([250 / 255, 182 / 255, 1 / 255, 1.]),
}


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


def color_visualization(viz: MeshcatVisualizer,
                        assembly: ModuleAssembly,
                        color_map: Dict[Union[str, int, module_classification.ModuleType], np.ndarray] = None):
    """
    Adds colors to a visualized robot in viz, depending on module types.

    This method only works for visualizers that already contain a (colorless) visualization of the robot based on the
    provided assembly. It will then color the robot according to the provided color_map.
    :param viz: The visualizer to color
    :param assembly: The assembly (that can already be visualized in viz) to color
    :param color_map: A dictionary mapping module integer enumerated module types OR module IDs to colors. If color_map
      maps module IDs to colors, this method first tries to interpret them as custom module IDs matching the internal
      representation in the assembly. If this fails, it will interpret them as original module IDs.
    """
    # We can't perform a proper equality check but at least we can perform some necessary conditions
    if not (tuple(viz.model.names) == tuple(assembly.robot.model.names)
            and set(g.name for g in viz.visual_model.geometryObjects)
            == set(g.name for g in assembly.robot.visual.geometryObjects)):
        raise ValueError("The provided visualizer and assembly do not match."
                         "Make sure to call viz = assembly.robot.visualize() first.")
    if color_map is None:
        color_map = DEFAULT_COLOR_MAP
    if all(isinstance(k, str) for k in color_map.keys()):
        cmap = color_map
        by_type = False
    else:
        cmap = {int(k): v for k, v in color_map.items()}  # Make sure that both, enumerated types and integers, work
        by_type = True
    viz_cmap = dict()
    for id_custom, id_original, m in zip(assembly.internal_module_ids,
                                         assembly.original_module_ids,
                                         assembly.module_instances):
        if by_type:
            key = int(module_classification.get_module_type(m))
        elif id_custom in cmap:
            key = id_custom
        else:
            key = id_original
        if key in cmap:
            color = cmap[key]
        else:
            color = None

        for body in m.bodies:
            viz_name = '.'.join(body.id)
            viz_cmap[viz_name] = color

    for go in assembly.robot.visual.geometryObjects:
        stem = '.'.join(go.name.split('.')[:-1])
        if stem in viz_cmap:
            viz.viewer[viz.viewerVisualGroupName + '/' + go.name].set_property(u'color', viz_cmap[stem].tolist())
        else:
            logging.warning(f"Could not find color for robot geometry {go.name}.")


def animation(robot: 'Robot.PinRobot', q: np.ndarray, dt: float,  # pragma: no cover # noqa: F821
              visualizer: Union[pin.visualize.MeshcatVisualizer, MeshcatVisualizerWithAnimation] = None
              ) -> MeshcatVisualizerWithAnimation:
    """
    Creates an animation of a robot movement.

    :param robot: The robot to animate
    :param q: The joint angles to animate
    :param dt: The time step between frames
    :param visualizer: If given, the movie will be generated in the existing visualizer
    """
    if q.shape[1] != robot.dof:
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


def center_camera(viz: MeshcatVisualizer,
                  around: Union[np.ndarray, List[float], Tuple[float, float, float]]):
    """
    Centers the camera of the viewer window s.t. it targets the origin (if it is not already centered otherwise).

    It is placed behind the "around" placement, s.t. any object placed there is shown prominently. The camera will, as
    common in robotics, be placed above the object.

    :param viz: The viewer to center (e.g. from `MeshcatVisualizer`)
    :param around: The point to look at
    """
    p = np.asarray(around).squeeze()
    rotated_transform = spatial.rotX(-np.pi / 2)
    if p.shape != (3,):
        raise ValueError("The 'around' point must be a 3D vector.")
    place_camera = np.eye(4)
    place_camera[:2, 3] = around[:2] + np.sign(around[:2]) * 1.5
    place_camera[2, 3] = around[2] + 1.5
    viz.viewer["/Cameras/default/rotated/<object>"].set_transform(rotated_transform @ place_camera)


def clear_visualizer(visualizer: MeshcatVisualizer):
    """
    Clears the viewer window, i.e. leaving it open but removing all objects that are typically set.

    Usually, visualizer.clean() alone should be enough, but at some occasions this does not work. This method has been
    tested to work in all cases so far.

    :param visualizer: The visualizer to clear
    """
    visualizer.clean()
    visualizer.viewer.window.send(meshcat.commands.Delete('visuals'))
    visualizer.viewer.window.send(meshcat.commands.Delete('collisions'))
    visualizer.viewer.window.send(meshcat.commands.Delete('meshcat'))
    visualizer.viewer.window.send(meshcat.commands.Delete('<object>'))
    center_camera(visualizer, [1.5, 0, 0])


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
    viz.viewer[name + '_arr_body'].set_transform(base_transform.homogeneous)

    viz.viewer[name + '_arr_head'].set_object(head, material)
    head_transform = placement @ rot @ spatial.homogeneous(translation=[0, -.5 * head_length, 0])
    viz.viewer[name + '_arr_head'].set_transform(head_transform.homogeneous)


def place_sphere(viz: MeshcatVisualizer, name: str, radius: float, placement: Transformation,
                 material: meshcat.geometry.Material = meshcat.geometry.MeshBasicMaterial()):
    """
    Place a sphere with name, radius and color at a specific placement.

    :param viz: MeshcatVisualizer instance to draw into (viz.viewer if you have a MeshcatVisualizer)
    :param name: Unique object name within the visualizer
    :param radius: Sphere radius in meters
    :param placement: Defines placement of the sphere.
    :param material: material used to render this sphere
    """
    viz.viewer[name].set_object(meshcat.geometry.Sphere(radius), material)
    viz.viewer[name].set_transform(placement.homogeneous)


def save_visualizer_scene(viz: MeshcatVisualizer,
                          filename: Union[Path, str],
                          overwrite: bool = False,
                          camera_transform: Optional[Transformation] = None
                          ):
    """
    Saves the current visualizer scene as static html file.

    Other than save_visualizer_screenshot(), this method does not require the viewer to be open.

    :param viz: MeshcatVisualizer instance
    :param filename: Filename to save the screenshot to. If it does not end with .html, it will be appended.
    :param overwrite: If True, the file will be overwritten if it already exists.
    :param camera_transform: If provided, the camera will be placed at this transformation before saving the scene.
    """
    filename = Path(filename)
    if not filename.suffix == '.html':
        filename = filename.with_suffix('.html')
    if filename.exists() and not overwrite:
        raise FileExistsError("File {} already exists.".format(filename))
    if camera_transform is not None:
        viz.viewer['/Cameras/default'].set_transform(camera_transform.homogeneous)

    scene = viz.viewer.static_html()
    with open(filename, 'w') as f:
        f.write(scene)


def save_visualizer_screenshot(viz: MeshcatVisualizer,
                               filename: Union[Path, str],
                               overwrite: bool = False,
                               timeout_seconds: int = 1,
                               camera_transform: Optional[Transformation] = None,
                               open_at_timeout: bool = False,
                               crop_to_content: bool = False
                               ):
    """
    Takes a screenshot of the current visualizer viewer and saves it to a png file.

    The get_image() method for a MeshcatVisualizer only works when the viewer is open. If not, it stalls forever, which
    is why a timeout is used here.

    :param viz: MeshcatVisualizer instance
    :param filename: Filename to save the screenshot to. If it does not end with .png, it will be appended.
    :param overwrite: If True, the file will be overwritten if it already exists.
    :param timeout_seconds: Timeout (integers only). If getting the screenshot is not successfull during this time,
        nothing will be saved. This is commonly the case when the viewer is not open.
    :param camera_transform: If provided, this will be the '/Cameras/default' setting used to take the screenshot.
    :param open_at_timeout: If True, the viewer will be opened if the timeout is reached. If so, this method will be
        called again, trying to take the screenshot a second time
    :param crop_to_content: If true, crops the image to the bounding box of the content. This only affects the image
        if the background is transparent (a filled background counts as content).
    """
    filename = map2path(filename)
    if not filename.suffix == '.png':
        filename = filename.with_suffix('.png')
    if filename.exists() and not overwrite:
        raise FileExistsError("File {} already exists".format(filename))
    if camera_transform is not None:
        viz.viewer['/Cameras/default'].set_transform(camera_transform.homogeneous)

    success = False  # noqa: F841  (linting does not recognize the timeout decorator)
    img = None  # noqa: F841  (linting does not recognize the timeout decorator)
    with timeout(timeout_seconds):
        img = viz.viewer.get_image()
        success = True

    if not success:
        # After we timeout, the zmq socket is broken - we need to create a new one
        viz.viewer.window.connect_zmq()

    if not success and open_at_timeout:
        viz.viewer.open()
        save_visualizer_screenshot(viz, filename, overwrite, timeout_seconds, camera_transform, False,
                                   crop_to_content=crop_to_content)

    if success:
        if crop_to_content:
            img = img.crop(img.getbbox())
        img.save(str(filename))


def place_billboard(viz: MeshcatVisualizer, text: str, name: str, placement: TransformationConvertable,
                    text_color: str = 'black', background_color: str = 'transparent', scale: float = 1.,
                    base_width: int = 100, font_size: int = 32, super_sample: float = 4.):
    """
    Add a billboard (2D text) to the visualization.

    Note needs to install additional dependency with `pip install timor-python.[viz]`.

    :param viz: MeshcatVisualizer instance to draw into (viz.viewer if you have a MeshcatVisualizer).
    :param text: Text to be displayed (does not support newline but many of the unicode characters, s.a. emoticons).
    :param name: Unique object name within the visualizer (overwrites existing object with same name).
    :param placement: Defines placement of the billboard.
    :param text_color: Color of the text (CSS4 color name or hex code, e.g. #abcdef).
    :param background_color: Color of the background (CSS4 color name or hex code, e.g. #abcdef).
    :param scale: Scaling factor for the text and billboard size.
    :param base_width: (Max) width of the billboard in pixels; will be smaller if text is shorter.
    :param font_size: Font size in pixels.
    :param super_sample: How much to increase font_size and base_width and reduce scale to get crisper text.
    """
    if not hasattr(meshcat.geometry, 'BillboardObject'):
        logging.warning("Missing dependencies, text visualization not possible."
                        "Please install with 'pip install timor-python.[viz]'")
        return

    if len(text) > 30:
        logging.warning("Text very long and might be squished.")
    if text_color not in matplotlib.colors.CSS4_COLORS and not re.match(r'^#[a-f0-9]{6}$', text_color) and \
            text_color not in {'transparent'}:
        logging.warning(
            f"Text color {text_color} not valid CSS4 color name or hex code (format #abcdef); may be ignored.")
    if background_color not in matplotlib.colors.CSS4_COLORS and not re.match(r'^#[a-f0-9]{6}$', background_color) and \
            background_color not in {'transparent'}:
        logging.warning(f"Background color {background_color} not valid CSS4 color name or hex code (format #abcdef);"
                        f" may be ignored.")
    if "\n" in text:
        logging.warning("Text contains newline character that will be ignored.")

    text_geom = meshcat.geometry.BillboardObject(text,
                                                 base_width=base_width * super_sample,
                                                 size=font_size * super_sample,
                                                 global_scale=scale * 0.01 / super_sample,
                                                 text_color=text_color,
                                                 background_color=background_color)
    viz.viewer[name].set_object(text_geom)
    viz.viewer[name].set_transform(Transformation(placement).homogeneous)


def plot_time_series(times: Sequence[float], data: Sequence[Tuple[np.ndarray, str]],
                     marker: Dict[float, Tuple[str, str]] = None, additional_subplots: int = 0,
                     show_figure: bool = False, subplot_kwargs: Optional[Dict] = None) \
        -> plt.Figure:
    """
    Helper to plot time series data

    :param times: Time (x-value) for the sampled data at N steps
    :param data: Timeseries, each containing a NxM array of N samples of M-dimensional data at each time in times and
      an axis label for this data; creates one subplot per timeseries
    :param marker: A dictionary where keys are times to draw a marking line with label and color at
    :param additional_subplots: leave space for more subplots
    :param show_figure: Call show on returned figure
    :param subplot_kwargs: Kwargs handed to subplots, including pyplot.figure kwargs
      (see https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html)
    :return: figure containing all these
    """
    if marker is None:
        marker = {}

    data = tuple(data)
    default_subplot_kwargs = {"sharex": "row"}
    default_subplot_kwargs.update({} if subplot_kwargs is None else subplot_kwargs)
    f, axes = plt.subplots(len(data) + additional_subplots, 1, **default_subplot_kwargs)
    if not isinstance(axes, Iterable):
        axes = (axes,)

    for idx, d in enumerate(data):
        axes[idx].plot(times, d[0])
        axes[idx].set(ylabel=d[1])

    axes[-1].set(xlabel="Time t [s]")

    time_frame = np.max(times) - np.min(times)

    for time, marker in marker.items():
        name, color = marker
        for ax in axes:
            ax.axvline(time, c=color)
            ax.text(time + 0.01 * time_frame,  # Slight offset from line
                    .9, name, c=color, transform=ax.get_xaxis_transform())

    plt.tight_layout()
    if show_figure:
        plt.show()

    return f
