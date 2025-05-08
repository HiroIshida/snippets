from dataclasses import dataclass
import time
import numpy as np
from pddlstream.utils import read
from skrobot.model.primitives import Cylinder
from pddlstream.algorithms.meta import solve
from pddlstream.language.constants import Exists, And, Not
from pddlstream.language.constants import PDDLProblem
from pddlstream.language.stream import DEBUG, SHARED_DEBUG
from pddlstream.language.generator import from_test, from_fn, from_gen_fn


domain_pddl = read("domain.l")
constant_map = {}
stream_pddl = read("stream.l")
stream_map = {}

@dataclass
class Object:
    name: str

@dataclass
class Grasp:
    name: str

@dataclass
class Traj:
    name: str

@dataclass
class Conf:
    name: str

@dataclass
class Pose:
    name: str

def sample_grasp(obj):
    return (Grasp("sampled"),)

def sample_path_to_grasp(obj, pose, grasp):
    return (Traj("reach-gras"), Conf("sampled"))  # dummy

def sample_path_relocate(obj, pose1, grasp, q1, pose2, q2):
    return (Traj("relocate"),)  # dummy

def sample_pose(obj):
    while True:
        yield (Pose("sampled"),)


stream_map["sample-grasp"] = from_fn(sample_grasp)
stream_map["sample-path-to-grasp"] = from_fn(sample_path_to_grasp)
stream_map["sample-path-relocate"] = from_fn(sample_path_relocate)
stream_map["sample-pose"] = from_gen_fn(sample_pose)

q_start = Conf("start")
cylinder_pose = Pose("init")
cylinder = Object("cylinder")

init = []
init.append(("TypeConf", q_start))
init.append(("AtConf", q_start))
init.append(("AtPose", cylinder, cylinder_pose))
init.append(("TypePose", cylinder_pose))
init.append(("IsGraspable", cylinder))
init.append(("IsHandEmpty",))
goal = ("DebugFlag",)

problem = PDDLProblem(domain_pddl, constant_map, stream_pddl, stream_map, init, goal)
from pyinstrument import Profiler
profiler = Profiler()
profiler.start()
solution = solve(problem, algorithm="adaptive", unit_costs=True, success_cost=float("inf")) 
profiler.stop()
print(profiler.output_text(unicode=True, color=True, show_all=False))
print(solution)
for item in solution.plan:
    print(item.name)
    print(item.args)
