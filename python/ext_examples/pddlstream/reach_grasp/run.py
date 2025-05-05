import time
import numpy as np
from pddlstream.utils import read
from skrobot.model.primitives import Cylinder
from pddlstream.algorithms.meta import solve
from pddlstream.language.constants import Exists, And
from pddlstream.language.constants import PDDLProblem
from pddlstream.language.stream import DEBUG, SHARED_DEBUG
from pddlstream.language.generator import from_test, from_fn, from_gen_fn

domain_pddl = read("domain.l")
constant_map = {}
stream_pddl = read("stream.l")
stream_map = {}


def sample_grasp_pose(obj):
    for i in range(5):
        yield (np.random.uniform(-np.pi, np.pi, 3),)
    return None


def sample_path_to_pose(q1, pose):
    for i in range(5):
        yield np.random.uniform(-np.pi, np.pi, 3), np.random.uniform(-np.pi, np.pi, 3)
    return None


stream_map["sample-grasp-pose"] = from_gen_fn(sample_grasp_pose)
stream_map["sample-path-to-pose"] = from_gen_fn(sample_path_to_pose)

q_start = np.array([0.0, 1.32, 1.40, -0.20, 1.72, 0.0, 1.66, 0.0])
cylinder = Cylinder(radius=0.1, height=0.2)

init = []
init.append(("AtConf", q_start))
init.append(("IsConf", q_start))
init.append(("IsGraspable", cylinder))
init.append(("IsHandEmpty",))

# goal = Exists(["?pose", "?obj", "?q"], And(("IsGrasp", "?pose", "?obj"), ("Kin", "?q", "?pose"), ("AtConf", "?q")))
goal = ("IsHolding", cylinder)
problem = PDDLProblem(domain_pddl, constant_map, stream_pddl, stream_map, init, goal)

print("start!")
ts = time.time()
solution = solve(problem, algorithm="adaptive", unit_costs=True, success_cost=float("inf")) 
te = time.time() - ts
print(f"Time: {te} seconds")
print(solution)
