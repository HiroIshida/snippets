import matplotlib.pyplot as plt
import numpy as np
from mohou.types import VectorBase, AngleVector, ElementDict, EpisodeData, ElementSequence, EpisodeBundle
from mohou.dataset import AutoRegressiveDatasetConfig
from mohou.file import get_project_path, create_project_dir
from mohou.script_utils import create_default_logger, train_lstm
from mohou.encoder import VectorIdenticalEncoder 
from mohou.encoding_rule import EncodingRule
from dataset import TargetPosition
from mohou.types import AngleVector, TerminateFlag
from mohou.trainer import TrainConfig, TrainCache
from mohou.model.lstm import PBLSTM, PBLSTMConfig

project_path =  get_project_path("pblstm_test")
bundle = EpisodeBundle.load(project_path)

tcache = TrainCache.load_latest(project_path, PBLSTM)
torch_params = tcache.best_model.parametric_bias_list
params = np.array([p.detach().numpy() for p in torch_params])
steps = [len(episode) for episode in bundle._episode_list]
fig, ax = plt.subplots()
plt.scatter(steps, params)
plt.xlabel("episode length")
plt.ylabel("pb value")
plt.show()
