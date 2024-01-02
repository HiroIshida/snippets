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


@dataclass
class Observation:
    fly_dist: float
    proximity: float
    action: RayCastAction


def ray_marching(pts_starts, direction_arr_unit, f_sdf) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pts_starts = copy.deepcopy(pts_starts)
    ray_tips = pts_starts
    n_point = len(pts_starts)

    frying_dists = np.zeros(n_point)
    proximity = np.ones(n_point) * np.inf

    for _ in range(20):
        dists = f_sdf(ray_tips)
        proximity = np.minimum(proximity, dists)
        ray_tips += dists * direction_arr_unit
        frying_dists += dists
    return ray_tips, proximity, frying_dists


def observe(sdf: Callable[[np.ndarray], np.ndarray],
            ray: RayCastAction) -> Observation:
    pts_start = np.expand_dims(ray.start, axis=0)
    direction_arr_unit = np.expand_dims(ray.direction / np.linalg.norm(ray.direction), axis=0)
    pts_tips, proximity, fly_dist = ray_marching(pts_start, direction_arr_unit, sdf)
    return Observation(fly_dist=fly_dist[0], proximity=proximity[0], action=ray)


class HypothesisPruner:
    link: Link
    margin: float

    def __init__(self, link: Link, margin: float):
        # assert link has sdf property
        assert hasattr(link, 'sdf')
        assert link.sdf is not None
        self.link = link
        self.margin = margin

    def is_unlikely(self,
                   hypo: Coordinates,
                   obs: Observation,
                   ) -> bool:
        is_hit_miss = obs.fly_dist > obs.action.cast_dist
        if is_hit_miss:  # the information is not enough to prune
            return False
        else:
            self.link.newcoords(hypo)
            est_obs = observe(self.link.sdf, obs.action)
            is_est_hit = est_obs.fly_dist < obs.action.cast_dist
            if is_est_hit:
                diff = np.abs(est_obs.fly_dist - obs.fly_dist)
                return diff > self.margin
            else:
                obvious_hit_miss = est_obs.proximity > self.margin
                if obvious_hit_miss:
                    return True
                else:
                    return False

    def prune(self, hypotheses: List[Coordinates], obs: Observation) -> List[Coordinates]:
        return [hypo for hypo in hypotheses if not self.is_unlikely(hypo, obs)]


def instantiate_box(co: Coordinates) -> Box:
    box = Box([0.05, 0.1, 0.05], face_colors=[255, 0, 0, 100])
    box.newcoords(co)
    return box


if __name__ == "__main__":
    co_true = Coordinates(pos = [0.02, 0.02, 0.0], rot=[0.2, 0.0, 0.0])
    box_true = Box([0.05, 0.1, 0.05], face_colors=[0, 255, 0, 255], with_sdf=True)
    box_true.newcoords(co_true)
    pruner = HypothesisPruner(copy.deepcopy(box_true), margin=0.01)

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

    actions = []
    action = RayCastAction(start=np.array([-0.5, 0.0, 0.0]), direction=np.array([1.0, 0.0, 0.0]), cast_dist=1.0)
    actions.append(action)
    obs = observe(box_true.sdf, action)
    H = pruner.prune(H, obs)

    action = RayCastAction(start=np.array([-0.5, 0.04, 0.0]), direction=np.array([1.0, 0.0, 0.0]), cast_dist=1.0)
    actions.append(action)
    obs = observe(box_true.sdf, action)
    H = pruner.prune(H, obs)

    action = RayCastAction(start=np.array([-0.5, -0.03, 0.0]), direction=np.array([1.0, 0.0, 0.0]), cast_dist=1.0)
    actions.append(action)
    obs = observe(box_true.sdf, action)
    H = pruner.prune(H, obs)

    box_list = [instantiate_box(hypo) for hypo in H]
    v = TrimeshSceneViewer()
    v.add(box_true)
    for action in actions:
        v.add(action.line)
    for box in box_list:
        v.add(box)
    v.show()
    time.sleep(1000)
