from dataclasses import dataclass
from typing import Optional

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
from mohou.types import (
    AngleVector,
    ElementDict,
    EpisodeData,
    MultiEpisodeChunk,
    RGBImage,
)


@dataclass
class Everything:
    chunk: MultiEpisodeChunk
    dataset_intact: MarkovControlSystemDataset
    dataset_nonintact: MarkovControlSystemDataset
    model: ControlModel
    obs_rule: EncodingRule
    ctrl_rule: EncodingRule

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
        dataset = MarkovControlSystemDataset.from_chunk(
            chunk, ctrl_rule, obs_rule, config=config, diff_as_control=True
        )
        return dataset

    @classmethod
    def load(cls) -> "Everything":
        chunk = MultiEpisodeChunk.load()
        dataset_intact = cls.create_dataset(chunk.get_intact_chunk())
        dataset_nonintact = cls.create_dataset(chunk.get_not_intact_chunk())
        tcache = TrainCache.load(None, ControlModel)
        assert tcache.best_model is not None
        model = tcache.best_model
        return cls(
            chunk,
            dataset_intact,
            dataset_nonintact,
            model,
            cls.create_obs_rule(),
            cls.create_ctrl_rule(chunk),
        )


@dataclass
class MarkovPropagator:
    model: ControlModel
    ctrl_rule: EncodingRule
    obs_rule: EncodingRule
    ctrl_pre: Optional[np.ndarray] = None

    @classmethod
    def construct(cls, every: Everything) -> "MarkovPropagator":
        return cls(every.model, every.ctrl_rule, every.obs_rule)

    def get_next(self, edict: ElementDict) -> ElementDict:
        obs = self.obs_rule.apply(edict)
        ctrl = self.ctrl_rule.apply(edict)

        if self.ctrl_pre is None:
            self.ctrl_pre = ctrl
            ctrl_diff = np.zeros(self.ctrl_rule.dimension)
        else:
            ctrl_diff = ctrl - self.ctrl_pre

        ctrl_diff_tensor = torch.from_numpy(ctrl_diff).float()
        obs_tensor = torch.from_numpy(obs).float()

        inp_sample = torch.concat((ctrl_diff_tensor, obs_tensor))
        out_sample = self.model.layer(inp_sample.unsqueeze(dim=0)).squeeze()
        edict_out = self.obs_rule.inverse_apply(out_sample.detach().numpy())
        return edict_out


def compute_diff(every: Everything):
    for triplet in every.dataset_intact:
        ctrl_inp, obs_inp, obs_out = triplet

        inp = torch.concat((ctrl_inp, obs_inp))
        assert inp.ndim == 1
        obs_pred = every.model.layer(inp.unsqueeze(dim=0)).squeeze(dim=0)
        l = nn.MSELoss()(obs_pred, obs_out)
        print(l)

        obs_pred_np = obs_pred.detach().numpy()
        obs_out_np = obs_out.detach().numpy()
        edict_pred = every.obs_rule.inverse_apply(obs_pred_np)
        edict_out = every.obs_rule.inverse_apply(obs_out_np)

        rgb_pred = edict_pred[RGBImage]
        rgb_out = edict_out[RGBImage]

        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(rgb_pred.numpy())
        axes[1].imshow(rgb_out.numpy())
        plt.show()


def propagation_test(every: Everything):
    prop = MarkovPropagator.construct(every)
    chunk = every.chunk.get_intact_chunk()
    episode: EpisodeData = chunk[0]

    for i in range(50):
        edict = episode.get_elem_dict(i)
        edict_next_gt = episode.get_elem_dict(i + 1)
        edict_next_pred = prop.get_next(edict)

        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(edict_next_gt[RGBImage].numpy())
        axes[1].imshow(edict_next_pred[RGBImage].numpy())
        plt.show()


data = Everything.load()
propagation_test(data)

# compute_diff(data)
