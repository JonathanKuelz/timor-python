import numpy as np
from numpy import pi

from timor import ModuleAssembly, ModulesDB


def main():
    print("This demo shows how to export a module assembly to an STL file, e.g., for 3D printing, editing, subtracting "
          "from 3D scenes, etc. This is done via trimesh which also offers additional editing, export, etc. "
          "functionality as stated in their documentation.")
    db = ModulesDB.from_name('IMPROV')
    a = ModuleAssembly.from_serial_modules(db, ['1', '21', '4', '21', '15', '22', '5', '23', '12'])
    a.robot.update_configuration(np.array((0, pi / 7, -2 * pi / 7, 0, -pi / 7, 0, pi / 10, 0)))
    a.robot.visualize()  # Check location/configuration
    tri = a.export_to_trimesh()
    file_name = "improv_long.stl"
    tri.export(file_name, "stl")
    print(f"Exported to {file_name}")
    print("You can now open this file in a 3D editor, 3D printer slicer, or similar, s.a., meshlab.")


if __name__ == '__main__':
    main()
