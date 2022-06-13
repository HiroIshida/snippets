if __name__ == "__main__":

    def task_process(task_class):

        from pathlib import Path
        from typing import List

        import numpy as np
        import tqdm
        from moviepy.editor import ImageSequenceClip

        from rlbench.action_modes.action_mode import MoveArmThenGripper
        from rlbench.action_modes.arm_action_modes import JointVelocity
        from rlbench.action_modes.gripper_action_modes import Discrete
        from rlbench.demo import Demo
        from rlbench.environment import Environment
        from rlbench.observation_config import ObservationConfig

        def rlbench_demo_to_rgb_seq(demo: Demo) -> List[np.ndarray]:
            seq = []
            for obs in demo:
                rgbs = []
                for camera_name in [
                    "overhead",
                    "left_shoulder",
                    "right_shoulder",
                    "front",
                    "wrist",
                ]:
                    rgbs.append(obs.__dict__[camera_name + "_rgb"])
                rgb = np.concatenate(rgbs, axis=1)
                seq.append(rgb)
            return seq

        obs_config = ObservationConfig()
        obs_config.set_all(True)

        env = Environment(
            action_mode=MoveArmThenGripper(
                arm_action_mode=JointVelocity(), gripper_action_mode=Discrete()
            ),
            obs_config=ObservationConfig(),
            headless=True,
        )
        env.launch()

        resolution = 112
        dirpath = Path("./rlbench_gif")
        dirpath.mkdir(exist_ok=True)

        print("executing task class named {}".format(task_class.__name__))
        task = env.get_task(task_class)

        mohou_episode_data_list = []

        demo = task.get_demos(amount=1, live_demos=True)[0]
        rgb_seq = rlbench_demo_to_rgb_seq(demo)
        clip = ImageSequenceClip([img for img in rgb_seq], fps=50)
        file_path = dirpath / (task_class.__name__ + ".gif")
        clip.write_gif(str(file_path), fps=50)

    import inspect
    import os
    from multiprocessing import Pool

    import rlbench.tasks
    from rlbench.backend.task import Task

    all_task_classes = []
    for attr in rlbench.tasks.__dict__.values():
        if not inspect.isclass(attr):
            continue
        if issubclass(attr, Task):
            all_task_classes.append(attr)

    p = Pool(int(os.cpu_count() * 0.5))
    p.map(task_process, all_task_classes)
