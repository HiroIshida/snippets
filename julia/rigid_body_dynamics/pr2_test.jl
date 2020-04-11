using RigidBodyDynamics
using MeshCatMechanisms, Blink
urdf_path = ENV["HOME"] * "/.skrobot/pr2_description/pr2.urdf"
const mech = parse_urdf(urdf_path)
state = MechanismState(mech)
rand_configuration!(state)
vis = MechanismVisualizer(mechanism, URDFVisuals(urdf_path))
