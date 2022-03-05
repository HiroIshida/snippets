from yacs.config import CfgNode

_config = CfgNode()
_config.rosbag = CfgNode()
_config.rosbag.topics = None

_config.train = CfgNode()
_config.train.epoch = None
_config.train.iteration = None

def get_cfg_defaults():
    return _config.clone()
