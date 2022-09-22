import tqdm
import numpy as np
from dataclasses import dataclass
from arhmm.propagator import Propagator
from typing import Any, Optional, TypeVar, Generic, List, Tuple


@dataclass
class SegmentationInfo:
    indices_phase_finish: List[int]

    @classmethod
    def init(cls, n_seq_len: int, n_phase: int) -> "SegmentationInfo":
        indices_list = np.array_split(list(range(n_seq_len)), n_phase)
        return cls([indices[-1] for indices in indices_list])

    @property
    def n_phase(self) -> int:
        return len(self.indices_phase_finish)

    def get_slice(self, phase: int) -> slice:
        assert phase < self.n_phase + 1

        indices_phase_start = [0] + [idx + 1 for idx in self.indices_phase_finish[:-1]]

        idx_start = indices_phase_start[phase]
        idx_finish = self.indices_phase_finish[phase]

        return slice(idx_start, idx_finish + 1)

    def get_phase(self, index: int):
        for phase, idx_finish in enumerate(self.indices_phase_finish):
            if index <= idx_finish:
                return phase
        assert False


class PropWrap(Propagator):

    def compute_loss(self, x: np.ndarray, x_next: np.ndarray) -> float:
        return -np.log(self.transition_prob(x, x_next))


def evaluate_loss(x_seq: np.ndarray, props: List[PropWrap], seg_info: SegmentationInfo) -> float:
    loss_total = 0.0
    for i in range(len(x_seq)-1):
        phase = seg_info.get_phase(i)
        prop = props[phase]
        x = x_seq[i]
        x_next = x_seq[i + 1]
        loss_total += prop.compute_loss(x, x_next)
    return loss_total


def find_best_partition(x_seq: np.ndarray, props: List[PropWrap]) -> Tuple[SegmentationInfo, float]:
    n_seq_len = len(x_seq)
    seg_info_candidates = []
    for i in range(n_seq_len-1):
        seg_info = SegmentationInfo([i, n_seq_len - 1])
        seg_info_candidates.append(seg_info)

    loss_list = [evaluate_loss(x_seq, props, seg_info) for seg_info in seg_info_candidates]
    idx_min = np.argmin(loss_list).item()
    return seg_info_candidates[idx_min], loss_list[idx_min]



def fit_parameter(x_seq_list: List[np.ndarray], seginfo_list: List[SegmentationInfo]) -> List[PropWrap]:
    n_phase = seginfo_list[0].n_phase
    assert len(x_seq_list) == len(seginfo_list)
    props = []
    for phase in range(n_phase): 
        xs_list = []
        for x_seq, seginfo in zip(x_seq_list, seginfo_list):
            bound = seginfo.get_slice(phase)
            xs_list.append(x_seq[bound])
        props.append(PropWrap.fit_parameter(xs_list))
    return props


def train(x_seq_list: List[np.ndarray], n_phase: int = 2) -> Tuple[List[SegmentationInfo], List[PropWrap]]:
    # create initial seginfo
    seginfo_list = []
    for x_seq in x_seq_list:
        seginfo = SegmentationInfo.init(len(x_seq), n_phase)
        seginfo_list.append(seginfo)

    props = fit_parameter(x_seq_list, seginfo_list)
    for i in range(3):

        # expectation
        loss_total = 0.0
        seginfo_list = []
        for x_seq in tqdm.tqdm(x_seq_list):
            seginfo, loss = find_best_partition(x_seq, props)
            seginfo_list.append(seginfo)
            loss_total += loss
        print(loss_total)

        # maximization
        props = fit_parameter(x_seq_list, seginfo_list)
    return seginfo_list, props


if __name__ == "__main__":
    import torch
    import numpy as np
    from mohou.types import EpisodeBundle
    from mohou.trainer import TrainCache
    from mohou.model.autoencoder import VariationalAutoEncoder
    from mohou.model.lstm import LSTM
    from mohou.types import RGBImage, AngleVector
    from mohou.default import create_default_encoding_rule, create_default_propagator, load_default_image_encoder
    from mohou.dataset.sequence_dataset import AutoRegressiveDataset, AutoRegressiveDatasetConfig
    from mohou.file import get_project_path
    from pathlib import Path
    from bunsetsu.split import SplitDescription, SplitDescriptionBundle
    from bunsetsu.file import get_segmentation_path

    pp = get_project_path("pr2_tidyup_dish")
    bundle = EpisodeBundle.load(pp)
    xs_list = []
    for episode in bundle:
        seq = episode.get_sequence_by_type(AngleVector)
        xs = np.array([vec for vec in seq])
        xs_list.append(xs)
    seginfo_list, _ = train(xs_list)

    sd_list = []
    for seginfo in seginfo_list:
        sd = SplitDescription.from_end_indices(seginfo.indices_phase_finish)
        sd_list.append(sd)

    sp = get_segmentation_path(pp, "test2")
    sd_bundle = SplitDescriptionBundle(sd_list)
    sd_bundle.dump(sp)
