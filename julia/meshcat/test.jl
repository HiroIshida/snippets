using RigidBodyDynamics
import MeshCat
using Revise
using MeshCatMechanisms, MechanismGeometries, Blink

urdf_path = "./pr2_description/pr2.urdf"
uvis = URDFVisuals(urdf_path; package_path=[""])

mech = parse_urdf(urdf_path)
state = MechanismState(mech)
rand_configuration!(state)

function angle_vector!(vis, av)
    active_jointname_list = ["l_shoulder_pan_joint",
                        "l_shoulder_lift_joint",
                        "l_upper_arm_roll_joint",
                        "l_elbow_flex_joint",
                        "l_forearm_roll_joint",
                        "l_wrist_flex_joint",
                        "l_wrist_roll_joint",
                        "r_shoulder_pan_joint",
                        "r_shoulder_lift_joint",
                        "r_upper_arm_roll_joint",
                        "r_elbow_flex_joint",
                        "r_forearm_roll_joint",
                        "r_wrist_flex_joint",
                        "r_wrist_roll_joint",
                        "torso_lift_joint",
                        "head_pan_joint",
                        "head_tilt_joint"]
    robot = mech
    for (angle, joint_name) in zip(av, active_jointname_list)
        joint = findjoint(robot, joint_name)
        set_configuration!(vis, joint, angle)
    end
end

vis = MechanismVisualizer(mech, uvis)
av = [0.3 0.3 0 0 0 0 0 0 0 0 0 0 0 0 0.3 0 0]
angle_vector!(vis, av)

