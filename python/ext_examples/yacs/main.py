from config import get_cfg_defaults

cfg = get_cfg_defaults()
cfg.rosbag.merge_from_file('./rosbag.yaml')
cfg.train.merge_from_file('./train.yaml')
print(cfg)

