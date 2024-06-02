import numpy as np
import pinocchio as pin
import tempfile

def load_robot(joint_type = "revolute"):
    urdf_string = f"""
<?xml version="1.0"?>
<robot name="dummy">

  <link name="base_link"/>

  <link name="link2">
    <inertial>
      <origin rpy="0 0 0" xyz="0 0 0" />
      <mass value="2.3311" />
      <inertia ixx="0.0019" ixy="0.0" ixz="0.0" iyy="0.0045" iyz="0.0" izz="0.0047" />
    </inertial>
  </link>

  <joint name="joint" type="{joint_type}">
    <origin rpy="0 0 0" xyz="0.0 0 0" />
    <parent link="base_link" />
    <child link="link2" />
    <axis xyz="1 0 0" />
    <limit effort="76.94" velocity="1.571" lower="-100.0" upper="100.0" />
  </joint>

</robot>"""
    with tempfile.NamedTemporaryFile(mode="w") as urdf_file:

        with open(urdf_file.name, "w") as f:
            f.write(urdf_string)

        robot = pin.RobotWrapper.BuildFromURDF(
            filename=urdf_file.name,
            package_dirs=None,
            root_joint=pin.JointModelFreeFlyer())
    return robot

for joint_type in ["revolute", "continuous"]:
    robot = load_robot(joint_type)
    if joint_type == "revolute":
        q = np.array([0, 0, 0, 0, 0, 0, 1, 0])
    else:
        q = np.array([0, 0, 0, 0, 0, 0, 1, 1, 0])  # [1, 0] <= [cos(0), sin(0)]
    Ag = pin.computeCentroidalMap(robot.model, robot.data, q)
    print(Ag[3:6, 3:6])
