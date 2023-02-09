#!/usr/bin/env python3
# Author: Jonathan Külz
# Date: 08.06.22
import xml.etree.ElementTree as ET

import numpy as np
import pinocchio as pin

from timor import Geometry
from timor.utilities.spatial import mat2euler
from timor.utilities.transformation import Transformation


def from_geometry(geometry: Geometry.Geometry,
                  which: str,
                  name: str,
                  link_frame: np.ndarray = Transformation.neutral().homogeneous) -> ET.Element:
    """
    Takes a Geometry and turns it into a URDF geometry

    :param geometry: Geometry instance to transform
    :param which: Either <visual> or <collision>, decides which tag is used
    :param name: The name of the robot geometry
    :param link_frame: The relative position of the body frame regarding the previous joint. URDF does not know the
        concept of a distinct body/link frame, so we have to change the positioning accordingly.
    """
    if which not in ('visual', 'collision'):
        raise ValueError(f"Invalid Geometry tag {which}. Must be visual or collision.")

    if isinstance(geometry, Geometry.EmptyGeometry):
        raise ValueError("Cannot write empty geometries to URDF")

    def att2str(att: any) -> str:
        if isinstance(att, str):
            return att
        elif isinstance(att, (np.ndarray, list, tuple)):
            return ' '.join(map(str, att))
        else:
            return str(att)

    urdf = ET.Element(which, {'name': name})
    origin = link_frame @ geometry.placement.homogeneous
    ET.SubElement(urdf, 'origin', {
        'xyz': ' '.join(map(str, origin[:3, 3])),
        'rpy': ' '.join(map(str, mat2euler(origin[:3, :3], 'xyz')))
    })

    geometry_element = ET.SubElement(urdf, 'geometry')
    tag, attributes = geometry.urdf_properties
    ET.SubElement(geometry_element, tag, {key: att2str(val) for key, val in attributes.items()})
    return urdf


def from_pin_inertia(i: pin.Inertia, link_frame: np.ndarray = Transformation.neutral()):
    """
    Takes a pin Inertia and returns a URDF <Inertial> element

    :param i: The original inertia data from a body
    :param link_frame: The relative position of the body frame regarding the previous joint. URDF does not know the
        concept of a distinct body/link frame, so we have to change the positioning accordingly.
    """
    inertial = ET.Element('inertial')
    orientation = link_frame[:3, :3]
    origin = orientation @ i.lever + link_frame[:3, 3]
    ET.SubElement(inertial, 'origin', {'xyz': ' '.join(map(str, origin)),
                                       'rpy': ' '.join(map(str, mat2euler(orientation, 'xyz')))
                                       })
    ET.SubElement(inertial, 'mass', {'value': str(i.mass)})
    str_idx = 'xyz'
    ET.SubElement(inertial, 'inertia', {
        f'i{str_idx[n]}{str_idx[m]}': str(i.inertia[n, m]) for n, m in
        ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))
    })
    return inertial
