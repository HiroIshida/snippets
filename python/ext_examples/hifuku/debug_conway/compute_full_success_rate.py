from hifuku.script_utils import load_library
from hifuku.domain import FetchConwayJailInsert

domain = FetchConwayJailInsert
tt = domain.task_type
lib = load_library(domain, "cuda", postfix="0.1")
# lib.jit_compile(batch_predictor=False)
solver = domain.solver_type.init(domain.solver_config)
rate_history = []

solve_count = 0
for it in range(2000):
    task = tt.sample()
    infres = lib.infer(task)
    if infres.cost <= lib.max_admissible_cost:
        solver.setup(task.export_problem())
        res = solver.solve(infres.init_solution)
        if res.traj is not None:
            solve_count += 1
    print(f"iter = {it}")
    print(f"rate = {solve_count / (it + 1)}")
    rate_history.append(solve_count / (it + 1))

import matplotlib.pyplot as plt 
plt.plot(rate_history)
plt.show()
