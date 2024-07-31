import numpy as np
from mohou_ros_utils.types import TimeStampedSequence
from mohou_ros_utils.rosbag import bag_to_synced_seqs
from mohou_ros_utils.interpolator import (
    AllSameInterpolationRule,
    NearestNeighbourMessageInterpolator,
)
from typing import Any, Optional, TypeVar, Generic, List, Tuple, Dict, Type
from mohou_ros_utils.conversion import RGBImageConverter
import rosbag
from pathlib import Path
from jsk_hironx_teleop.msg import FloatVector
from sensor_msgs.msg import CompressedImage
from mohou.types import AngleVector, EpisodeData, ElementSequence, EpisodeBundle
from tunable_filter.composite_zoo import HSVBlurCropResolFilter


# topic_names = ["/hironx_imitation/larm/robot_action", "/head_camera/rgb/image_raw/compressed"]
# seqs = bag_to_synced_seqs(bag, 0.2, rule=rule, topic_names=topic_names)

def convert_seq_to_mohou_types(seq: TimeStampedSequence):
    if seq.object_type == CompressedImage:
        conv = RGBImageConverter("/head_camera/rgb/image_raw/compressed")
        conv.image_filter = image_filter
        seq = [conv.apply(msg) for msg in seq.object_list]
        return seq
    elif seq.object_type == FloatVector:
        def apply(msg: FloatVector):
            # from the data, it seems that only 6 elements are used 
            # TODO(iwata) Am I right?
            return AngleVector(np.array(msg.data)[6:12])
        return [apply(msg) for msg in seq.object_list]
    else:
        assert False, "unsupported type"

def convert_seqs_to_episode(seqs: List[TimeStampedSequence]):
    elem_seqs = [ElementSequence(convert_seq_to_mohou_types(seq)) for seq in seqs]
    edata = EpisodeData.from_seq_list(elem_seqs)
    return edata


def bag_to_episode(bag_path: Path, hz):
    topic_names = ["/hironx_imitation/larm/robot_action", "/head_camera/rgb/image_raw/compressed"]
    bag = rosbag.Bag(bag_path)
    rule = AllSameInterpolationRule(NearestNeighbourMessageInterpolator)
    seqs = bag_to_synced_seqs(bag, 1.0 / hz, rule=rule, topic_names=topic_names)
    bag.close()
    return convert_seqs_to_episode(seqs)


def bags_to_bundle(bag_parent_dir: Path, hz) -> EpisodeBundle:
    lst = []
    for bag_path in bag_parent_dir.glob("*.bag"):
        print(f"processing {bag_path}")
        lst.append(bag_to_episode(bag_path, hz))
    bundle = EpisodeBundle.from_episodes(lst)
    return bundle

# episode = bag_to_episode(Path("~/Downloads/hiro_demo_2024-07-30-11-22-41.bag").expanduser())
pp = Path("~/.mohou/iwata").expanduser()
image_filter = HSVBlurCropResolFilter.from_yaml(pp / "image_config.yaml")  # global (dirty)
bundle = bags_to_bundle(pp / "rosbag", 5)
bundle.plot_vector_histories(AngleVector, pp, hz=5)
bundle.dump(pp, exist_ok=True)

image_bundle = bags_to_bundle(pp / "rosbag", 20)
image_bundle.dump(pp, exist_ok=True, postfix="autoencoder")
