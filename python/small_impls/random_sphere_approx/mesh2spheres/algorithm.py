import logging
from typing import Callable, List, Tuple

import numpy as np
import tqdm
from trimesh import Trimesh

logger = logging.getLogger(__name__)


def generate_inner_and_surface_points(
    mesh: Trimesh,
    sdf: Callable[[np.ndarray], np.ndarray],
    margin: float,
    n_inner: int,
    n_surface: int,
) -> Tuple[np.ndarray, np.ndarray]:
    logger.info("generating inner and surface points")
    vertices = np.array(mesh.vertices)
    lb, ub = np.min(vertices, axis=0), np.max(vertices, axis=0)
    lb_exteded = lb - margin * 2
    ub_exteded = ub + margin * 2

    inner_points = []
    while len(inner_points) < n_inner:
        cands = np.random.uniform(lb_exteded, ub_exteded, (n_inner, 3))
        filtered = list(cands[sdf(cands) < 0.0])
        inner_points.extend(filtered)
        print(len(inner_points))
    inner_points = np.array(inner_points[:n_inner])

    surface_points = []
    while len(surface_points) < n_surface:
        cands = np.random.uniform(lb_exteded, ub_exteded, (n_surface, 3))
        filtered = list(cands[np.logical_and(sdf(cands) > margin, sdf(cands) < margin * 1.1)])
        surface_points.extend(filtered)
    surface_points = np.array(surface_points[:n_surface])
    return inner_points, surface_points


def optimize(
    inner_points: np.ndarray,
    surface_points: np.ndarray,
    margin: float,
    n_max_iter: int,
    coverage_tolerance: float,
) -> List[Tuple[np.ndarray, float]]:

    assert coverage_tolerance > 0.0 and coverage_tolerance < 1.0

    logger.info("determining candidate radii")
    radius_list = []
    for p in inner_points:
        sqdists = np.sum((surface_points - p) ** 2, axis=1)
        min_sqdist = np.min(sqdists)
        min_dist = np.sqrt(min_sqdist)
        radius_list.append(min_dist)
    radii = np.array(radius_list)

    logger.info("precomputing boolean array predicting covering")
    n_inner = len(inner_points)
    bools_covering = np.zeros((n_inner, n_inner), dtype=bool)
    for i in tqdm.tqdm(range(n_inner)):
        p, r = inner_points[i], radii[i]
        sqdists = np.sum((inner_points - p) ** 2, axis=1)
        bools_covering[i] = sqdists < r**2

    logger.info("start greedy optimization")
    sphere_list = []
    bools_union_covering = np.zeros(n_inner, dtype=bool)
    for i in range(n_max_iter):
        np.sum(bools_union_covering)
        bools_union_covering_expected = bools_union_covering | bools_covering
        expected_cover_counts = np.sum(bools_union_covering_expected, axis=1)
        best_idx = np.argmax(expected_cover_counts)
        bools_union_covering = bools_union_covering_expected[best_idx]
        num_covering = np.sum(bools_union_covering)
        rate = num_covering / n_inner
        sphere_list.append((inner_points[best_idx], radii[best_idx]))
        logger.info(f"iter: {i}, rate: {rate} ({num_covering}/{n_inner})")
        if (1 - rate) < coverage_tolerance:
            break
    return sphere_list


def find_sphere_approximation(
    mesh: Trimesh,
    sdf: Callable[[np.ndarray], np.ndarray],
    margin: float,
    n_inner: int = 30000,
    n_surface: int = 10000,
    n_max_iter: int = 100,
    coverage_tolerance: float = 0.01,
) -> List[Tuple[np.ndarray, float]]:
    inner_points, surface_points = generate_inner_and_surface_points(
        mesh, sdf, margin, n_inner, n_surface
    )
    return optimize(inner_points, surface_points, margin, n_max_iter, coverage_tolerance)
