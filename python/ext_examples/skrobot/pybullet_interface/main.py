from lib import Environment

env = Environment(gui=True)
ret = env.solve_ik(env.co_grasp_pre, False, random_sampling=True)
print(f"finish first ik: {ret}")
ret = env.solve_ik(env.co_grasp, True, random_sampling=False)
print(f"finish second ik: {ret}")
env.grasp(False)
print(f"finish third grasp")
env.translate([0, 0, 0.05], True)
import time; time.sleep(1000)
