import xml.etree.ElementTree as ET
from pathlib import Path
import coacd
import trimesh
import ycb_utils

dir_path = Path("parts")
if not dir_path.exists():
    dir_path.mkdir()
    m = ycb_utils.load("019_pitcher_base")
    mesh = coacd.Mesh(m.vertices, m.faces)
    parts = coacd.run_coacd(mesh)
    for i, (V, F) in enumerate(parts):
        file_path = dir_path / f"part_{i}.obj"
        trimesh.Trimesh(V, F).export(file_path)

root = ET.Element('mujoco', attrib={'model': 'pitcher_decomposed'})
compiler = ET.SubElement(root, 'compiler', attrib={'angle': 'radian', 'coordinate': 'local'})

asset = ET.SubElement(root, 'asset')
for part_file in sorted(dir_path.glob("part_*.obj")):
    mesh_name = part_file.stem  # e.g. "part_0"
    ET.SubElement(asset, 'mesh', attrib={
        'file': part_file.as_posix(),
        'name': mesh_name
    })

worldbody = ET.SubElement(root, 'worldbody')

pitcher_body = ET.SubElement(worldbody, 'body', attrib={
    'name': 'pitcher_body',
    'pos': '0 0 0'
})

for part_file in sorted(dir_path.glob("part_*.obj")):
    mesh_name = part_file.stem
    ET.SubElement(pitcher_body, 'geom', attrib={
        'mesh': mesh_name,
        'name': f"{mesh_name}_geom",
        'type': 'mesh',
        'contype': '1',
        'conaffinity': '1'
    })

tree = ET.ElementTree(root)
xml_path = "pitcher_decomposed.xml"
tree.write(xml_path, encoding='utf-8', xml_declaration=True)
print(f"MuJoCo XML file generated: {xml_path}")
