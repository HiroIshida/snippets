import itertools
import numpy as np
import tqdm
from typing import Optional


def exact_max_cover(
        k,
        sets,
        costs: Optional[np.ndarray] = None,
        cost_max: Optional[float] = None) -> int:

    n_set = len(sets)
    best_score = -np.inf
    for index_tuple in tqdm.tqdm(itertools.combinations(range(n_set), k)):
        subsets = sets[np.array(index_tuple)]
        score = np.sum(np.any(subsets, axis=0))
        if cost_max is not None:
            subcosts = costs[np.array(index_tuple)]
            cost = np.sum(subcosts)
            if cost > cost_max:
                continue
        if best_score < score:
            best_score = score
    return best_score


def greedy_max_cover(
        k,
        sets: np.ndarray,
        costs: Optional[np.ndarray] = None,
        cost_max: Optional[float] = None) -> int:

    sets = list(sets)

    n_target = len(sets[0])
    is_covered = np.zeros(n_target, dtype=bool)

    solution_sets = []
    for _ in range(k):
        current_score = np.sum(is_covered)

        def gain(S) -> int:
            is_covered_cand = np.logical_or(is_covered, S)
            score_cand = np.sum(is_covered_cand)
            return score_cand - current_score

        max_index = max(range(len(sets)), key=lambda i: gain(sets[i]))

        S_finally = sets[max_index]
        gain = gain(S_finally)
        is_covered = np.logical_or(is_covered, S_finally)
        solution_sets.append(sets.pop(max_index))

        current_cost = 

    score = np.sum(np.any(solution_sets, axis=0))
    return score


np.random.seed(0)
n_target = 400
n_element = 20
p_parameter = 0.2
k = 8
cost_max = k * 0.4
sets = np.array([np.random.choice(a=[False, True], size=(n_target,), p=[1-p_parameter, p_parameter]) for _ in range(n_element)])
costs = np.random.rand(n_target)
best_score = exact_max_cover(k, sets, costs, None)
print(best_score)
best_score = greedy_max_cover(k, sets, costs, cost_max)
print(best_score)
