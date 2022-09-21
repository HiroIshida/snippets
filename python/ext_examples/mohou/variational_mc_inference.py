import torch
import numpy as np
import tqdm
from mohou.types import AngleVector, EpisodeBundle
from mohou.trainer import TrainCache
from mohou.model.lstm import LSTM
from mohou.file import get_project_path
from mohou.default import create_default_propagator, create_default_encoding_rule
from mohou.propagator import Propagator
from mohou.propagator import Propagator
from mohou.model.third_party.variational_lstm import VariationalLSTM

pp = get_project_path("pr2_tidyup_dish")
bundle = EpisodeBundle.load(pp)

rule = create_default_encoding_rule(pp)
episode = bundle[5]
#episode = bundle.get_untouch_bundle()[0]

lstm = TrainCache.load(pp, LSTM, variational=True).best_model
vlstm: VariationalLSTM = lstm.lstm_layer  # type: ignore
prop = Propagator(lstm, rule, device=torch.device("cpu"))
#prop = Propagator(lstm, rule)

av_seq_list = []
n_mc = 30
for _ in tqdm.tqdm(range(n_mc)):
    prop.reset()
    av_list = []
    for i in range(len(episode)):
        edict = episode[i]
        if i==0:
            av_list.append(edict[AngleVector].numpy())
        prop.feed(edict)
        vlstm.randomize()
        pred = prop.predict(1)[0]
        av_list.append(pred[AngleVector].numpy())
    av_seq = np.array(av_list)
    av_seq_list.append(av_seq)
av_seq_arr = np.array(av_seq_list)

cov_list = []
det_list = []
for t in range(len(episode)):
    avs = av_seq_arr[:, t, :]
    cov =  np.cov(avs.T)
    cov_list.append(cov)
    det_list.append(np.linalg.det(cov))

import matplotlib.pyplot as plt
plt.plot(np.log(det_list[1:]))
plt.show()
