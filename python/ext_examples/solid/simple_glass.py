from pathlib import Path 
import tempfile
import argparse
import trimesh
from solid import cylinder, scad_render_to_file, translate
import numpy as np
import subprocess

def generate_glass_stl(thickness, height, outer_radius, filename='glass.stl'):
    segments = 30
    outer_cylinder = cylinder(h=height, r=outer_radius, segments=segments)

    inner_radius = outer_radius - thickness
    inner_cylinder = cylinder(h=height - thickness, r=inner_radius, segments=segments)

    inner_cylinder = translate([0, 0, thickness])(inner_cylinder)

    glass = outer_cylinder - inner_cylinder

    with tempfile.TemporaryDirectory() as td:
      file_path = Path(td) / "glass.scad"
      scad_render_to_file(glass, file_path)
      print(f"convert {file_path} to {filename}")
      subprocess.run(["openscad", "-o", filename, file_path])  # please apt install openscad

def visualize_stl(filename):
    mesh = trimesh.load_mesh(filename)
    mesh.show()

def main():
    parser = argparse.ArgumentParser(description="Generate a handle-less glass STL file.")
    parser.add_argument('--thickness', type=float, default=0.005, help="Thickness of the glass wall.")
    parser.add_argument('--height', type=float, default=0.1, help="Height of the glass.")
    parser.add_argument('--outer_radius', type=float, default=0.04, help="Outer radius of the glass.")
    parser.add_argument('--filename', type=str, default='glass.stl', help="Filename for the output STL file.")
    parser.add_argument('--visualize', action='store_true', help="Visualize the generated STL file.")
    
    args = parser.parse_args()
    
    generate_glass_stl(args.thickness, args.height, args.outer_radius, args.filename)
    
    if args.visualize:
        visualize_stl(args.filename)

if __name__ == "__main__":
    main()
