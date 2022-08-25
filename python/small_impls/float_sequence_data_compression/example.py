from pathlib import Path
import gzip
import os
from tempfile import TemporaryDirectory
import numpy as np
from mohou.types import EpisodeBundle, DepthImage
from mohou.file import get_project_path

def mark_adler_algorithm(depth_seq, threshold = 1e-4):
    diffs = depth_seq[1:] - depth_seq[:-1]
    diffs[diffs < threshold] = 0.0
    ma_arr = np.array(list(diffs) + depth_seq[0])
    return ma_arr

pp = get_project_path("pybullet_reaching_RGB")
bundle = EpisodeBundle.load(pp)

tmp = bundle[0].get_sequence_by_type(DepthImage)

depth_arr = np.array([e.numpy() for e in tmp])
ma_arr = mark_adler_algorithm(depth_arr)

with TemporaryDirectory() as td:
    
    original_path = Path(td) / "original.npy"
    np.save(original_path, depth_arr)
    print(os.path.getsize(original_path))

    mark_alder_path = Path(td) / "ma.npz"
    np.savez_compressed(mark_alder_path, ma_arr)
    print(os.path.getsize(mark_alder_path))
