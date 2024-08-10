import numpy as np
import time
from rpbench.articulated.fetch.tidyup_table import TidyupTableTask
from skrobot.models import Fetch
from hifuku.domain import FetchTidyupTable
from hifuku.script_utils import load_library
from plainmp.robot_spec import FetchSpec
from plainmp.ompl_solver import OMPLSolver, OMPLSolverConfig
from plainmp.utils import set_robot_state
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# task creation
table_pos = np.array([0.7, 0.0])
bbox_param_list = [
    np.array([0.8, 0.2, 0.4, 0.1, 0.1, 0.2]),
    np.array([0.6, -0.4, 0.0, 0.1, 0.2, 0.25]),
]
target_xyz_yaw = np.array([0.6, 0.0, 0.85, 0.0])
task = TidyupTableTask.from_semantic_params(table_pos, bbox_param_list, target_xyz_yaw)

# you can solve it without trajectory library
problem = task.export_problem()
solver = OMPLSolver(OMPLSolverConfig())
ret = solver.solve(problem)
assert ret.traj is not None
print(f"naive: solved in {1000 * ret.time_elapsed} [ms]")

# load library (jit compilation makes inference 20x faster)
jit_compile = False
lib = load_library(FetchTidyupTable, "cuda", postfix="0.1")
if jit_compile:
    lib.jit_compile()
    # warm up jit requires two calls (why two...?). This takes over 10 sec.
    lib.infer(TidyupTableTask.sample())
    lib.infer(TidyupTableTask.sample())
    print("finished jit compile")

ts = time.time()
infer_res = lib.infer(task)
print(f"elapsed time for inference: {time.time() - ts}")

if infer_res.cost > lib.max_admissible_cost:
    print(f"sampled task seems to infesible")
else:
    print(f"sampled task seems to feasiblem, so solve it")
    conf = OMPLSolverConfig(n_max_ik_trial=1)
    solver = OMPLSolver(conf)
    ret = solver.solve(task.export_problem(), infer_res.init_solution)
    assert ret.traj is not None
    print(f"learned: solved in {1000 * ret.time_elapsed} [ms]")

    # visualize
    fs = FetchSpec()
    robot = Fetch()
    robot.reset_pose()
    v = task.create_viewer()
    v.add(robot)
    v.show()
    time.sleep(2)
    for q in ret.traj:
        set_robot_state(robot, fs.control_joint_names, q)
        v.redraw()
        time.sleep(0.3)
    time.sleep(100)
