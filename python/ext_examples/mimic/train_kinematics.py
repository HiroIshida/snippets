import argparse
import os
import pybullet_data
import torch

from mimic.dataset import KinematicsDataset
from mimic.models import KinemaNet, DenseConfig
from mimic.trainer import train
from mimic.trainer import Config
from mimic.trainer import TrainCache
from mimic.scripts.utils import split_with_ratio
from mimic.scripts.utils import create_default_logger

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-activation', type=str, default='relu')
    args = parser.parse_args()
    activation = args.activation
    robot_name = 'kuka'

    pbdata_path = pybullet_data.getDataPath()
    urdf_path = os.path.join(pbdata_path, 'kuka_iiwa', 'model.urdf')
    joint_names = ['lbr_iiwa_joint_{}'.format(idx+1) for idx in range(7)]
    link_names = ['lbr_iiwa_link_7']

    project_name = 'kinematics'
    logger = create_default_logger(project_name, 'kinemanet_{}'.format(robot_name))

    dataset = KinematicsDataset.from_urdf(urdf_path, joint_names, link_names)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = KinemaNet(device, dataset.meta_data, DenseConfig(200, 6, activation))

    ds_train, ds_valid = split_with_ratio(dataset)
    tcache = TrainCache[KinemaNet](project_name, KinemaNet, cache_postfix='_' + robot_name)
    config = Config(batch_size=1000, n_epoch=3000) 
    train(model, ds_train, ds_valid, tcache=tcache, config=config)
