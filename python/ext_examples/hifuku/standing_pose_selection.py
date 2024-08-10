import tqdm
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

task_list = []
x_pos_offset = np.linspace(-0.1, 0.1, 5)
y_pos_offset = np.linspace(-0.5, 0.5, 10)
xy_pairs = [(x, y) for x in x_pos_offset for y in y_pos_offset]
# xy_pairs = [(0., 0.0)]

for x_pos, y_pos in tqdm.tqdm(xy_pairs, desc="task generation"):
    table_pos = np.array([0.7 + x_pos, 0.0 + y_pos])
    bbox_param_list = [np.array([0.5 + x_pos, +0.3 + y_pos, 0.0, 0.2, 0.1, 0.25])]
    target_xyz_yaw = np.array([0.7 + x_pos, 0.4 + y_pos, 0.85, -0.3])
    task = TidyupTableTask.from_semantic_params(table_pos, bbox_param_list, target_xyz_yaw)
    task_list.append(task)

jit_compile = False  # please set True in the actual demo
lib = load_library(FetchTidyupTable, "cuda", postfix="0.1")
if jit_compile:
    lib.jit_compile()
    lib.infer(TidyupTableTask.sample())
    lib.infer(TidyupTableTask.sample())
    print("JIT compile done")

infer_res_list = []
for task in tqdm.tqdm(task_list, desc="inference"):
    infer_res = lib.infer(task)
    infer_res_list.append(infer_res)
expected_iters = np.array([res_infer.cost for res_infer in infer_res_list])
feasibilities = np.array(expected_iters) <= lib.max_admissible_cost
if np.any(feasibilities):
    print("Found feasible task")
    idx_best = np.argmin(expected_iters)
    task_best = task_list[idx_best]
    conf = OMPLSolverConfig(n_max_ik_trial=1)
    solver = OMPLSolver(conf)
    ret = solver.solve(task_best.export_problem(), infer_res_list[idx_best].init_solution)
    assert ret.traj is not None
    print(f"learned: solved in {1000 * ret.time_elapsed} [ms]")

    # visualize
    fs = FetchSpec()
    robot = Fetch()
    robot.reset_pose()
    v = task_best.create_viewer()
    v.add(robot)
    v.show()
    time.sleep(2)
    for q in ret.traj:
        set_robot_state(robot, fs.control_joint_names, q)
        v.redraw()
        time.sleep(0.3)
    time.sleep(100)
