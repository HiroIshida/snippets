import time
from pathlib import Path
from robot_descriptions.robotiq_2f85_mj_description import PACKAGE_PATH

import mujoco
from mujoco import mjx
from jax import lax
import jax
from mujoco_xml_editor import MujocoXmlEditor

hand_xml_path = Path(PACKAGE_PATH) / "2f85.xml"
editor = MujocoXmlEditor.load(hand_xml_path)
editor.add_ground()
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
    def batched_step(vel):
        mjx_data = mjx.make_data(mjx_model)
        mjx_data = step_n_times(mjx_model, mjx_data, 300)
        return mjx_data.qpos[0]

    fn = jax.vmap(batched_step)
    zeros = jax.numpy.zeros((3000,))
    result = fn(zeros)
    print("compiled")

    ts = time.time()
    result = fn(zeros)
    print("time per rollouts", (time.time() - ts) / 3000)
else:
    ts = time.time()
    data = mujoco.MjData(model)
    for _ in range(300):
        mujoco.mj_step(model, data)
    print("time per rollouts", (time.time() - ts))
