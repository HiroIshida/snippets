import bpy
import pathlib
import trimesh

from blendpred.utils.mesh import create_mesh_from_pydata


def create_mesh_from_file(
        scene: bpy.types.Scene,
        path: pathlib.Path,
        mesh_name: str,
        object_name: str,
        use_smooth: bool = True) -> bpy.types.Object:

    mesh = trimesh.load_mesh(str(path))
    V = mesh.vertices
    F = mesh.faces
    return create_mesh_from_pydata(scene, V, F, mesh_name, object_name, use_smooth=use_smooth)
