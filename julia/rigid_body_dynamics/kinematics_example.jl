using RigidBodyDynamics

function main()
    urdf_path = ENV["HOME"] * "/.skrobot/fetch_description/fetch.urdf"
    mech = parse_urdf(urdf_path)
    state = MechanismState(mech)
    joint_list = joints(mech)
    body_list = bodies(mech)
    link_names = ["l_gripper_finger_link", "r_gripper_finger_link", "wrist_flex_link", "wrist_roll_link", "shoulder_lift_link", "upperarm_roll_link"];
    end_body_list = RigidBody[]
    for name in link_names
        body = findbody(mech, name)
        push!(end_body_list, body)
    end
    points = [Point3D(default_frame(body), 0, 0, 0) for body in end_body_list]
    frames = [default_frame(body) for body in end_body_list]

    world = root_frame(mech)
    function bench_mark()
        for i in 1:1000000
            zero_configuration!(state)
            for frame in frames
                tf = relative_transform(state, frame, world)
            end
            """if you just want to get point
            for point in points
                coords = transform(state, point, world)
            end
            """
        end
    end
    println("start benchmarking")
    @time bench_mark()
    return relative_transform(state, frames[1], world)
end
tf = main()
