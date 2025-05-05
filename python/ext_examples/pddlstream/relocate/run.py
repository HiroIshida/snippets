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

class Traj: pass


def sample_grasp(obj):
    return (0.0,)

def sample_path_to_pose_from_home(pose):
    return (Traj, pose)  # dummy

def derive_grasp_coords(obj, pose, grasp):
    return (pose,)

def sample_path_relocate(obj, pose, grasp, co, q1, q2):
    return (Traj,)


stream_map["sample-grasp"] = from_fn(sample_grasp)
stream_map["sample-path-to-pose-from-home"] = from_fn(sample_path_to_pose_from_home)
stream_map["derive-grasp-coords"] = from_fn(derive_grasp_coords)
stream_map["sample-path-relocate"] = from_fn(sample_path_relocate)

q_start = 0
cylinder_pose = 1
cylinder = "cylinder"

init = []
init.append(("TypeConf", q_start))
init.append(("AtConf", q_start))
init.append(("AtPose", cylinder, cylinder_pose))
init.append(("TypePose", cylinder_pose))
init.append(("IsGraspable", cylinder))
init.append(("IsHandEmpty",))

# goal = And(Not(("IsHolding", cylinder)), ("IsInRegion", cylinder_pose))
# goal = ("IsInRegion", cylinder_pose)
# goal = Exists(("?pose"), And(("AtPose", cylinder, "?pose"), ("IsInRegion", "?pose")))
# goal = Exists(("?pose"), ("AtPose", cylinder, "?pose"))
# goal = ("IsHolding", cylinder)
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
