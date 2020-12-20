import skrobot
rm1 = skrobot.models.PR2()
rm2 = skrobot.models.urdf.RobotModelFromURDF(urdf_file=skrobot.data.pr2_urdfpath())

jl1 = set([j.name for j in rm1.joint_list])
jl2 = set([j.name for j in rm2.joint_list])

ll1 = set([l.name for l in rm1.link_list])
ll2 = set([l.name for l in rm2.link_list])
