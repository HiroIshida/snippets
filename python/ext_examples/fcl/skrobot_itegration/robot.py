import numpy as np
import pybullet as pb
from typing import List, Dict, Tuple, Set, Union
from skmp.robot.utils import PybulletRobot
from skmp.robot.fetch import FetchConfig
from skrobot.models.fetch import Fetch
from skrobot.model.link import Link
from skrobot.model.robot_model import RobotModel
from skrobot.utils.urdf import mesh_simplify_factor
from skrobot.coordinates.math import matrix2quaternion, wxyz2xyzw
import argparse
from trimesh import Trimesh

import fcl
from fcl import BVHModel, Transform, CollisionObject, DistanceResult, DistanceRequest


class FCLCollisionManager:
    link_name_to_model: Dict[str, BVHModel] = {}

    def __init__(self, robot: RobotModel):
        link_name_to_model = {}
        for link in robot.link_list:
            link: Link
            collmesh: Trimesh = link.collision_mesh
            if collmesh is not None:
                # extract verts and tries from collmesh
                verts = collmesh.vertices
                tris = collmesh.faces
                m = fcl.BVHModel()
                m.beginModel(len(verts), len(tris))
                m.addSubModel(verts, tris)
                m.endModel()
                col_obj = CollisionObject(m)
                link_name_to_model[link.name] = col_obj
        self.link_name_to_model = link_name_to_model

    def set_pose(self, link_name: str, position: np.ndarray, rotmat: np.ndarray):
        tf = Transform(rotmat, position)
        self.link_name_to_model[link_name].setTransform(tf)

    def reflect_skrobot(self, robot: RobotModel):
        for link_name in self.link_name_to_model.keys():
            link: Link = robot.__dict__[link_name]
            rot = link.worldrot()
            pos = link.worldpos()
            self.set_pose(link_name, pos, rot)

    def detect_collision(self, link1: str, link2: str) -> bool:
        req = DistanceRequest()
        req.enable_signed_distance = True
        req.gjk_solver_type = 1
        res = DistanceResult()
        fcl.distance(self.link_name_to_model[link1],
                     self.link_name_to_model[link2],
                     req, res)
        print(res.min_distance)



skfetch = Fetch()
skfetch.reset_pose()
skfetch.shoulder_lift_joint.joint_angle(0.5)  # intensionaly make collision
man = FCLCollisionManager(skfetch)
man.reflect_skrobot(skfetch)
ret = man.detect_collision('base_link', 'wrist_flex_link')
print(ret)  
ret = man.detect_collision('base_link', 'head_pan_link')
print(ret)
