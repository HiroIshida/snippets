import tqdm
from typing import List, Optional
import time
import numpy as np
from mohou.types import VectorBase, AngleVector, ElementDict, EpisodeData, ElementSequence, EpisodeBundle, MetaData
from mohou.file import get_project_path, create_project_dir
from skrobot.models.pr2 import PR2
from skrobot.coordinates import Coordinates
from skrobot.utils.urdf import mesh_simplify_factor


class TargetPosition(VectorBase):
    pass

joint_names = [
    "r_shoulder_pan_joint",
    "r_shoulder_lift_joint",
    "r_upper_arm_roll_joint",
    "r_elbow_flex_joint",
    "r_forearm_roll_joint",
    "r_wrist_flex_joint",
    "r_wrist_roll_joint",
]

def get_angle_vector(pr2: PR2) -> np.ndarray:
    q = np.array([pr2.__dict__[jn].joint_angle() for jn in joint_names])
    return q


def set_angle_vector(pr2: PR2, jnames: List[str], q: np.ndarray) -> None:
    assert len(jnames) == len(q)
    for jn, angle in zip(jnames, q):
        pr2.__dict__[jn].joint_angle(angle)


def create_single_episode(pr2: PR2, target: np.ndarray, n_split: int = 20) -> Optional[np.ndarray]:
    pr2.reset_pose()
    q_init = get_angle_vector(pr2)
    target_co = Coordinates(pos=target)
    ret = pr2.inverse_kinematics(
        target_co,
        link_list=pr2.rarm.link_list,
        move_target=pr2.rarm_end_coords,
        rotation_axis=False,
        stop=100,
    )
    q_terminal = get_angle_vector(pr2)
    # create trajectory by splitting q_init and q_terminal into n_split
    diff = (q_terminal - q_init) / (n_split - 1)
    qs = np.array([q_init + i * diff for i in range(n_split)])
    return qs


def create_dataset(n_data) -> EpisodeBundle:
    with mesh_simplify_factor(0.1):
        pr2 = PR2()
    episode_list = []

    pbar = tqdm.tqdm(total=n_data)
    while len(episode_list) < n_data:
        n_split = np.random.randint(20, 100)
        b_min = np.array([0.5, -0.5, 0.5])
        b_max = np.array([0.8, -0.0, 0.8])
        target = np.random.rand(3) * (b_max - b_min) + b_min
        qs = create_single_episode(pr2, target, n_split=n_split)
        if qs is None:
            continue

        avs = [AngleVector(q) for q in qs]
        targets = [TargetPosition(target) for _ in range(n_split)]
        episode = EpisodeData.from_seq_list([ElementSequence(avs), ElementSequence(targets)])
        episode_list.append(episode)
        pbar.update(1)

    bundle = EpisodeBundle.from_episodes(episode_list)
    return bundle


if __name__ == "__main__":
    bundle = create_dataset(100)
    create_project_dir("pblstm_test")
    bundle.dump(get_project_path("pblstm_test"), exist_ok=True)

    # from skrobot.viewers import TrimeshSceneViewer
    # v = TrimeshSceneViewer()
    # v.add(pr2)
    # v.show()
    # for q in qs:
    #     set_angle_vector(pr2, joint_names, q)
    #     time.sleep(0.5)
    # time.sleep(1000)
