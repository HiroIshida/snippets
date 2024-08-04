import time

from plainmp.ompl_solver import OMPLSolver, OMPLSolverConfig
from plainmp.robot_spec import FetchSpec
from skmp.robot.utils import set_robot_state
from skrobot.models.fetch import Fetch

from hifuku.domain import FetchTidyupTable
from hifuku.script_utils import load_library

domain = FetchTidyupTable
task_type = domain.task_type
task = task_type.sample()
lib = load_library(domain, "cuda", postfix="0.1")
# lib.jit_compile()

conf = OMPLSolverConfig(n_max_satisfaction_trial=1, n_max_call=10000)
solver = OMPLSolver(conf)
fs = FetchSpec()

ts = time.time()
infer_res = lib.infer(task)
print(f"infer time: {time.time() - ts}")
print(f"Cost: {infer_res.cost}, Max admissible cost: {lib.max_admissible_cost}")
if infer_res.cost > lib.max_admissible_cost:
    print(f"sampled task seems to infesible")
else:
    print(f"sampled task seems to feasiblem, so solve it")
    ret = solver.solve(task.export_problem(), infer_res.init_solution)
    assert ret.traj is not None
    print(f"elapsed time: {ret.time_elapsed}")
    v = task.create_viewer()
    fetch = Fetch()
    v.add(fetch)
    v.show()
    time.sleep(0.5)
    for q in ret.traj:
        set_robot_state(fetch, fs.control_joint_names, q)
        v.redraw()
        time.sleep(0.3)
    time.sleep(10000.0)
