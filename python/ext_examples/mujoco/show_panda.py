import numpy as np
import mujoco
import mujoco_viewer
from robot_descriptions.loaders.mujoco import load_robot_description

model = load_robot_description("panda_mj_description")
# model = mujoco.MjModel.from_xml_path('humanoid.xml')
data = mujoco.MjData(model)
viewer = mujoco_viewer.MujocoViewer(model, data)

# gripper pos is 255 because it is tendon driven and control range is 0-255
desired_qpos1 = np.array([0.0, -0.5, 0.0, -1.5, 0.0, 1.0, 0.7, 255])
desired_qpos2 = np.array([0.0, -0.1, 0.0, -1.5, 0.0, 1.0, 0.7, 0])
desired_qpos3 = np.array([-1.0, -0.1, 0.0, -1.5, 0.0, 1.0, 0.7, 255])
desired_list = [desired_qpos1, desired_qpos2, desired_qpos3]
current_index = 0
desired = desired_list[current_index]

kp = 100
kd = 10

input("press enter to start the simulation")

for _ in range(10000):
    if viewer.is_alive:
        current_qpos = data.qpos[:8]
        current_qvel = data.qvel[:8]
        pos_error = desired - current_qpos
        vel_error = -current_qvel
        if np.linalg.norm(pos_error[:7]) < 0.1:
            if current_index < len(desired_list) - 1:
                current_index += 1
                desired = desired_list[current_index]
        control_input = kp * pos_error + kd * vel_error
        data.ctrl[:8] = control_input
        mujoco.mj_step(model, data)
        viewer.render()
    else:
        break

viewer.close()
