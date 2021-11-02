import argparse
import sys
import typing
from typing import Union

import torch

from mimic.trainer import Config
from mimic.trainer import TrainCache
from mimic.trainer import train
from mimic.datatype import AbstractDataChunk
from mimic.datatype import ImageDataChunk
from mimic.datatype import ImageCommandDataChunk
from mimic.dataset import AutoRegressiveDataset
from mimic.dataset import BiasedAutoRegressiveDataset
from mimic.dataset import FirstOrderARDataset
from mimic.models import LSTMConfig
from mimic.models import LSTM
from mimic.models import BiasedLSTM
from mimic.models import DenseConfig
from mimic.models import DenseProp
from mimic.models import ImageAutoEncoder
from mimic.models import BiasedDenseProp
from mimic.models import DeprecatedDenseProp

from mimic.scripts.utils import split_with_ratio
from mimic.scripts.utils import create_default_logger
from mimic.scripts.utils import query_yes_no
from mimic.scripts.train_propagator import prepare_trained_image_chunk

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', type=int, default=1)
    args = parser.parse_args()
    mode = args.mode

    project_name = 'kuka_reaching'
    logger = create_default_logger(project_name, 'propagator_mode{}'.format(mode))
    chunk: ImageCommandDataChunk = prepare_trained_image_chunk(project_name)
    dataset = BiasedAutoRegressiveDataset.from_chunk(chunk)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if mode==0:
        dataset = AutoRegressiveDataset.from_chunk(chunk)
        prop_model = DenseProp(device, dataset.n_state, DenseConfig())
    elif mode==1:
        dataset = AutoRegressiveDataset.from_chunk(chunk)
        prop_model = LSTM(device, dataset.n_state, LSTMConfig())
        tcache = TrainCache[LSTM](project_name, LSTM)
    elif mode==2:
        dataset = BiasedAutoRegressiveDataset.from_chunk(chunk)
        prop_model = BiasedLSTM(device, dataset.n_state, dataset.n_bias, LSTMConfig())
        tcache = TrainCache[BiasedLSTM](project_name, BiasedLSTM)
    elif mode==3:
        dataset = FirstOrderARDataset.from_chunk(chunk)
        prop_model = DeprecatedDenseProp(device, dataset.n_state, DenseConfig())
        tcache = TrainCache[DeprecatedDenseProp](project_name, DeprecatedDenseProp)
    else:
        dataset = BiasedAutoRegressiveDataset.from_chunk(chunk)
        if mode==4:
            model_config = DenseConfig(200, 6)
        elif mode==5:
            model_config = DenseConfig(200, 6, 'relu')
        elif mode==6:
            model_config = DenseConfig(200, 6, 'sigmoid')
        else:
            sys.exit()
        prop_model = BiasedDenseProp(device, dataset.n_state, dataset.n_bias, model_config)
        tcache = TrainCache[BiasedDenseProp](project_name, BiasedDenseProp)

    ds_train, ds_valid = split_with_ratio(dataset)
    train(prop_model, ds_train, ds_valid, tcache=tcache, config=Config(n_epoch=3000))
