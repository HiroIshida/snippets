from pathlib import Path
from dataclasses import dataclass


def create_objfile_string(width, length, height, path: Path) -> str:
    objstring = """
mtllib {}
o Cube
""".format(path.name)
    for x in (width * 0.5, -width * 0.5):
        for y in (length * 0.5, -length * 0.5):
            for z in (height * 0.5, -height * 0.5):
                string = "v {} {} {}\n".format(x, y, z)
                objstring += string
    objstring += """
vt 0.625000 0.500000
vt 0.625000 0.750000
vt 0.875000 0.750000
vt 0.875000 0.500000
vt 0.375000 0.750000
vt 0.375000 1.000000
vt 0.625000 1.000000
vt 0.375000 0.000000
vt 0.375000 0.250000
vt 0.625000 0.250000
vt 0.625000 0.000000
vt 0.125000 0.500000
vt 0.125000 0.750000
vt 0.375000 0.500000
vn 0.0000 -1.0000 0.0000
vn 0.0000 0.0000 1.0000
vn -1.0000 0.0000 0.0000
vn 0.0000 1.0000 0.0000
vn 1.0000 0.0000 0.0000
vn 0.0000 0.0000 -1.0000
usemtl Material
s off
f 1/1/1 3/2/1 7/3/1 5/4/1
f 4/5/2 8/6/2 7/7/2 3/2/2
f 8/8/3 6/9/3 5/10/3 7/11/3
f 6/12/4 8/13/4 4/5/4 2/14/4
f 2/14/5 4/5/5 3/2/5 1/1/5
f 6/9/6 2/14/6 1/1/6 5/10/6
"""
    return objstring


def create_mtlfile_string(image_path: Path) -> str:
    string = """
# Blender MTL File: 'None'
# Material Count: 1

newmtl Material
Ns 323.999994
Ka 1.000000 1.000000 1.000000
Kd 0.800000 0.800000 0.800000
Ks 0.500000 0.500000 0.500000
Ke 0.000000 0.000000 0.000000
Ni 1.450000
d 1.000000
illum 2
map_Kd {}
""".format(str(image_path.expanduser().resolve()))
    return string


if __name__ == '__main__':
    obj_path = Path("~/tmp/dummy.obj").expanduser()
    mtl_path = Path("~/tmp/dummy.mtl").expanduser()
    objstring = create_objfile_string(10, 10, 1, obj_path)
    mtlstring = create_mtlfile_string(Path("./Wood081_1K_Color.jpg"))

    with obj_path.open(mode="w") as f:
        f.write(objstring)
    with mtl_path.open(mode="w") as f:
        f.write(mtlstring)
    #print(objstring)
    #print(mtlstring)
