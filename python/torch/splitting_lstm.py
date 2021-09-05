"""
Show that sequencial call of RNN and batch call will yield the same result
"""
import torch
from torch import nn

torch.manual_seed(0) 
n_dim = 3
n_hid = 10
n_seq = 100
model = nn.LSTM(n_dim, n_hid, batch_first=True)

# batch propagation
seq = torch.rand(1, n_seq, n_dim)
out_batch, hc_tuple_batch = model(seq, None)

# mini-batch propagation
mini_seqs = seq.split(30, dim=1)
hc_tuple_minimbatch = None
for mini_seq in mini_seqs:
    out, hc_tuple_minimbatch = model(mini_seq, hc_tuple_minimbatch)

# sequencial propagation
hc_tuple = None
out_batch_emurate = []
for i in range(n_seq):
    x = seq[:, torch.tensor([i]), :]
    out, hc_tuple = model(x, hc_tuple)
    out_batch_emurate.append(out)
out_batch_emurate = torch.cat(out_batch_emurate, 1)

# compare
assert torch.all(torch.isclose(out_batch, out_batch_emurate))
for hc_tuple_testing in [hc_tuple, hc_tuple_minimbatch]:
    for i in range(2):
        diff = hc_tuple_batch[i] - hc_tuple_testing[i]
        assert torch.all(torch.isclose(hc_tuple_batch[i], hc_tuple_testing[i]))
