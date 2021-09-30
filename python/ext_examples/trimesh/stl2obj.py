import trimesh
import os.path as osp

file_path = "/opt/ros/melodic/share/pr2_description/meshes/upper_arm_v0/forearm_roll_L.stl"

mesh = trimesh.load_mesh(file_path)
V = mesh.vertices
F = mesh.faces

def vert_line(arr):
    return "v {0} {1} {2}\n".format(arr[0], arr[1], arr[2])

def face_line(arr):
    return "f {0} {1} {2}\n".format(arr[0]+1, arr[1]+1, arr[2]+1)

with open("./tmp.obj", mode='w') as f:
    for vert in V:
        f.write(vert_line(vert))
    for face in F:
        f.write(face_line(face))


