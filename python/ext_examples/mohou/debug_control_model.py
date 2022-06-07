import argparse
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from mohou.dataset import MarkovControlSystemDataset, SequenceDatasetConfig
from mohou.encoder import VectorIdenticalEncoder
from mohou.encoding_rule import EncodingRule
from mohou.model import ControlModel, VariationalAutoEncoder
from mohou.model.markov import MarkoveModelConfig
from mohou.script_utils import create_default_logger
from mohou.setting import setting
from mohou.trainer import TrainCache, TrainConfig, train
from mohou.types import AngleVector, MultiEpisodeChunk, RGBImage


@dataclass
class Everything:
    chunk: MultiEpisodeChunk
    dataset_intact: MarkovControlSystemDataset
    dataset_nonintact: MarkovControlSystemDataset
    model: ControlModel
    obs_rule: EncodingRule

    @staticmethod
    def create_obs_rule():
        tcache = TrainCache.load(None, VariationalAutoEncoder)
        model = tcache.best_model
        assert model is not None
        f = model.get_encoder()
        obs_rule = EncodingRule.from_encoders([f])
        return obs_rule

    @staticmethod
    def create_ctrl_rule(chunk: MultiEpisodeChunk):
        n_av_dim = chunk.spec.type_shape_table[AngleVector][0]
        f = VectorIdenticalEncoder(AngleVector, n_av_dim)
        ctrl_rule = EncodingRule.from_encoders([f])
        return ctrl_rule

    @classmethod
    def create_dataset(cls, chunk: MultiEpisodeChunk) -> MarkovControlSystemDataset:
        obs_rule = cls.create_obs_rule()
        ctrl_rule = cls.create_ctrl_rule(chunk)
        config = SequenceDatasetConfig(n_aug=0)
        dataset = MarkovControlSystemDataset.from_chunk(chunk, ctrl_rule, obs_rule, config=config, diff_as_control=True)
        return dataset

    @classmethod
    def load(cls) -> 'Everything':
        chunk = MultiEpisodeChunk.load()
        dataset_intact = cls.create_dataset(chunk.get_intact_chunk())
        dataset_nonintact = cls.create_dataset(chunk.get_not_intact_chunk())
        tcache = TrainCache.load(None, ControlModel)
        assert tcache.best_model is not None
        model = tcache.best_model
        return cls(chunk, dataset_intact, dataset_nonintact, model, cls.create_obs_rule())


def compute_diff(everything: Everything):
    for triplet in everything.dataset_intact:
        ctrl_inp, obs_inp, obs_out = triplet

        inp = torch.concat((ctrl_inp, obs_inp))
        assert inp.ndim == 1
        obs_pred = everything.model.layer(inp.unsqueeze(dim=0)).squeeze(dim=0)
        l = nn.MSELoss()(obs_pred, obs_out)
        print(l)

        obs_pred_np = obs_pred.detach().numpy()
        obs_out_np = obs_out.detach().numpy()
        edict_pred = everything.obs_rule.inverse_apply(obs_pred_np)
        edict_out = everything.obs_rule.inverse_apply(obs_out_np)

        rgb_pred = edict_pred[RGBImage]
        rgb_out = edict_out[RGBImage]

        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(rgb_pred.numpy())
        axes[1].imshow(rgb_out.numpy())
        plt.show()
        




data = Everything.load()

compute_diff(data)
