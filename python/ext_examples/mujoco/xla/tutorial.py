import jax
import mujoco
from mujoco import mjx
import time
from jax import lax
import time

XML=r"""
<mujoco>
  <worldbody>
    <body>
      <freejoint/>
      <geom size=".15" mass="1" type="sphere"/>
    </body>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(XML)
data = mujoco.MjData(model)

# if I did on CPU
ts = time.time()
for _ in range(3000):
    mujoco.mj_step(model, data)
elapsed_cpu = time.time() - ts
print(f"Elapsed time: {elapsed_cpu}")

mjx_model = mjx.put_model(model)
print(jax.devices())

def step_once(model, data):
    return mjx.step(model, data)

def step_n_times(model, data, N):
    def body_fun(i, data):
        return step_once(model, data)
    return lax.fori_loop(0, N, body_fun, data)

@jax.jit
def batched_step(vel):
    mjx_data = mjx.make_data(mjx_model)
    qvel = mjx_data.qvel.at[0].set(vel)
    mjx_data = mjx_data.replace(qvel=qvel)
    mjx_data = step_n_times(mjx_model, mjx_data, 300)
    return mjx_data.qpos[0]

vel = jax.numpy.arange(0.0, 1.0, 0.001)
ts = time.time()
fn = jax.vmap(batched_step)
pos = fn(vel)
print(f"time to compile: {time.time() - ts}")

vel = jax.numpy.arange(0.0, 1.0, 0.001) * 2
ts = time.time()
pos = fn(vel)
elapsed_gpu = (time.time() - ts) / len(vel)
print("Elapsed time per solve: ", elapsed_gpu)
print(f"speedup: {elapsed_cpu / elapsed_gpu}")
