import skrobot
from skrobot.model import Sphere
from skrobot.sdf import UnionSDF

fetch = skrobot.models.Fetch()
fetch.reset_manip_pose()

def compute_joint_weights(joint_list, with_base=False):

    def measure_depth(joint):
        link = joint.parent_link
        depth = 1
        while link.parent_link is not None:
            link = link.parent_link
            depth += 1
        return depth
    joint_depth_list = map(measure_depth, joint_list)

    if with_base: 
        joint_depth_list = [d + 1 for d in joint_depth_list]
        joint_depth_list.extend([1, 1, 1])

    max_depth = max(joint_depth_list)
    joint_weight_list = [max_depth/(1.0*depth) for depth in joint_depth_list]
    return joint_weight_list

depth_list = compute_joint_weights(fetch.joint_list, with_base=True)
