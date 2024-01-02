import multiprocessing as mp
import numba
import tqdm
import copy
import time
import numpy as np
from dataclasses import dataclass
from typing import Any, Optional, TypeVar, Generic, List, Tuple, Dict, Type, Callable, Sequence
from skrobot.model.link import Link
from skrobot.coordinates import Coordinates
from skrobot.viewers import TrimeshSceneViewer
from skrobot.model.primitives import LineString, Box


@dataclass
class RayCastAction:
    start: np.ndarray
    direction: np.ndarray
    cast_dist: float

    @property
    def line(self) -> LineString:
        start = self.start
        end = self.start + self.direction * self.cast_dist
        line = LineString(np.array([start, end]))
        return line


def ray_marching(pts_starts, direction_arr_unit, f_sdf) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pts_starts = copy.deepcopy(pts_starts)
    ray_tips = pts_starts
    n_point = len(pts_starts)

    frying_dists = np.zeros(n_point)
    proximity = np.ones(n_point) * np.inf

    for _ in range(20):
        dists = f_sdf(ray_tips)
        proximity = np.minimum(proximity, dists)
        ray_tips += direction_arr_unit * dists[:, None]
        frying_dists += dists
    return ray_tips, proximity, frying_dists


def observe(sdf: Callable[[np.ndarray], np.ndarray],
            ray: RayCastAction) -> np.array:
    pts_start = np.expand_dims(ray.start, axis=0)
    direction_arr_unit = np.expand_dims(ray.direction / np.linalg.norm(ray.direction), axis=0)
    pts_tips, proximity, fly_dist = ray_marching(pts_start, direction_arr_unit, sdf)
    return np.array([fly_dist[0], proximity[0]])


class Blob:
    table: np.ndarray  # (hypo_idx, action_idx) -> Observation
    hypo_list: List[Coordinates]
    action_list: List[RayCastAction]

    def __init__(self, link: Link, hypo_list: List[Coordinates], action_list: List[RayCastAction], margin: float):
        link = copy.deepcopy(link)
        table = []
        for h in hypo_list:
            link.newcoords(h)
            pts_start = np.array([a.start for a in action_list])
            direction_arr_unit = np.array([a.direction / np.linalg.norm(a.direction) for a in action_list])
            pts_tips, proximity, fly_dist = ray_marching(pts_start, direction_arr_unit, link.sdf)
            sub_table = []
            for i, a in enumerate(action_list):
                obs = np.array([fly_dist[i], proximity[i]])
                sub_table.append(obs)
            table.append(sub_table)

        self.table = np.array(table).transpose(1, 0, 2)
        self.hypo_list = hypo_list
        self.action_list = action_list


class CacheUtilizedPruner:
    # This class becomes quite dirty in the process of optimization, but it is blazingly fast
    blob: Blob
    margin: float
    # indexed by (i_hypo, j_action). The cached data is stored as numpy array
    # np.ndarray of Observation object

    def __init__(self, blob: Blob, margin: float):
        self.blob = blob
        self.margin = margin

    def prune(self, obs: np.array, action_idx: int, hypo_indices: np.ndarray) -> np.ndarray:
        action_cast_dist = self.blob.action_list[action_idx].cast_dist
        is_hit_miss = obs[0] > action_cast_dist

        if is_hit_miss:
            return hypo_indices

        bools_survive = np.zeros(len(hypo_indices), dtype=bool)

        est_obs_arr = self.blob.table[action_idx, hypo_indices]
        is_est_hit_arr = est_obs_arr[:, 0] < action_cast_dist
        diff_arr = np.abs(est_obs_arr[:, 0] - obs[0])
        bools_survive[is_est_hit_arr & (diff_arr <= self.margin)] = True

        obvious_hit_miss_arr = est_obs_arr[:, 1] > self.margin
        bools_survive[~is_est_hit_arr & ~obvious_hit_miss_arr] = True
        return hypo_indices[bools_survive]


class GreedyPolicy:
    pruner: CacheUtilizedPruner
    pool: Optional[mp.Pool]

    def __init__(self,
                 pruner: CacheUtilizedPruner,
                 n_process: int = 4,
                 ):
        self.pruner = pruner
        if n_process > 1:
            self.pool = mp.Pool(mp.cpu_count(), initializer=self._initialize_worker_process, initargs=(pruner,))
        else:
            self.pool = None

    @staticmethod
    def _initialize_worker_process(pruner: CacheUtilizedPruner):
        global _G_pruner
        _G_pruner = pruner

    @staticmethod
    def _compute_score(i_action: int, hypo_indices: np.ndarray) -> float:
        global _G_pruner
        score_list = []
        for j_hypo in hypo_indices:
            obs_hypo = _G_pruner.blob.table[i_action, j_hypo]
            hypo_indices_new = _G_pruner.prune(obs_hypo, i_action, hypo_indices)
            score = len(hypo_indices) - len(hypo_indices_new)
            score_list.append(score)
        score_expectation = np.mean(score_list)
        return float(score_expectation)

    def __call__(self, hypo_indices: np.ndarray) -> int:
        # pool map 
        if self.pool is not None:
            score_list = self.pool.starmap(self._compute_score, [(i, hypo_indices) for i in range(len(self.pruner.blob.action_list))])
            i_best = np.argmax(score_list)
            return i_best
        else:
            i_best = None
            score_best = -np.inf
            for i in tqdm.tqdm(range(len(self.pruner.blob.action_list))):
                score_list = []
                for j in hypo_indices:
                    obs_hypo = self.pruner.blob.table[i, j]
                    hypo_indices_new = self.pruner.prune(obs_hypo, i, hypo_indices)
                    score = len(hypo_indices) - len(hypo_indices_new)
                    score_list.append(score)
                score_expectation = np.mean(score_list)
                if score_expectation > score_best:
                    score_best = score_expectation
                    i_best = i
            assert i_best is not None
            return i_best


def instantiate_box(co: Coordinates) -> Box:
    box = Box([0.05, 0.1, 0.05], face_colors=[255, 0, 0, 100])
    box.newcoords(co)
    return box


if __name__ == "__main__":
    np.random.seed(0)
    co_true = Coordinates(pos = [0.02, 0.02, 0.0], rot=[0.5, 0.0, 0.0])
    box_true = Box([0.05, 0.1, 0.05], face_colors=[0, 255, 0, 255], with_sdf=True)
    box_true.newcoords(co_true)

    n_action = 100
    action_set = []
    for y in np.linspace(-0.08, 0.08, n_action):
        action = RayCastAction(start=np.array([-0.5, y, 0.0]), direction=np.array([1.0, 0.0, 0.0]), cast_dist=1.0)
        action_set.append(action)

    n_hypo = 1000
    H = []
    for _ in range(n_hypo):
        box = Box([0.05, 0.1, 0.05], face_colors=[255, 0, 0, 100])
        hypo_b_min = np.array([-0.05, -0.05, -0.5])
        hypo_b_max = np.array([0.05, 0.05, 0.5])
        pos2d = np.random.uniform(hypo_b_min[:2], hypo_b_max[:2])
        pos3d = np.hstack([pos2d, 0.0])
        yaw = np.random.uniform(hypo_b_min[2], hypo_b_max[2])
        co = Coordinates(pos=pos3d, rot=[yaw, 0, 0])
        H.append(co)

    blob = Blob(box_true, H, action_set, margin=0.01)
    pruner = CacheUtilizedPruner(blob, margin=0.01)
    policy = GreedyPolicy(pruner, n_process=8)

    actions = []
    h_indices = np.arange(len(H))
    for _ in range(3):
        from pyinstrument import Profiler
        profiler = Profiler()
        profiler.start()
        a_idx = policy(h_indices)
        profiler.stop()
        print(profiler.output_text(unicode=True, color=True, show_all=True))
        actions.append(blob.action_list[a_idx])
        o = observe(box_true.sdf, blob.action_list[a_idx])
        h_indices = pruner.prune(o, a_idx, h_indices)
        if len(h_indices) == 0:
            break
        print(len(h_indices))

    box_list = [instantiate_box(H[h]) for h in h_indices]
    v = TrimeshSceneViewer()
    v.add(box_true)
    for action in actions:
        v.add(action.line)
    for box in box_list:
        v.add(box)
    v.show()
    time.sleep(1000)
