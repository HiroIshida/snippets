import time
import numpy as np
import ycb_utils
import mujoco
from mujoco import mjx
from mujoco_xml_editor import MujocoXmlEditor
import jax
from jax import lax


editor = MujocoXmlEditor.empty("test")
editor.add_sky()
editor.add_ground()
editor.add_light()
object_path = ycb_utils.resolve_path("035_power_drill")
pos = np.array([0.3, -0.025, 0.39])
editor.add_mesh(
    object_path,
    "drill",
    density=100,
    convex_decomposition=True,
    pos=pos,
    euler=np.array([1.54, np.pi, 0]),)
xmlstr = editor.to_string()
model = mujoco.MjModel.from_xml_string(xmlstr)

use_jax = True
if use_jax:
    mjx_model = mjx.put_model(model)

    def step_once(model, data):
        return mjx.step(model, data)

    def step_n_times(model, data, N):
        def body_fun(i, data):
            return step_once(model, data)
        return lax.fori_loop(0, N, body_fun, data)

    @jax.jit
    def batched_step(_):  # _ is a placeholder
        mjx_data = mjx.make_data(mjx_model)
        mjx_data = step_n_times(mjx_model, mjx_data, 300)
        return mjx_data.qpos[0]

    fn = jax.vmap(batched_step)
    zeros = jax.numpy.zeros((300,))
    result = fn(zeros)
    result = fn(zeros)
    print("compiled")

    ts = time.time()
    result = fn(zeros)
    print("time per rollouts", (time.time() - ts) / 300)
else:
    data = mujoco.MjData(model)
    ts = time.time()
    for _ in range(300):
        mujoco.mj_step(model, data)
    print("time per rollouts", time.time() - ts)

