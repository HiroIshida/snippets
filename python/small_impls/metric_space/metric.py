from dataclasses import dataclass
from math import acos, sqrt
from typing import List, Optional, Protocol

import numpy as np


def normalize(vec: np.ndarray) -> np.ndarray:
    return vec / np.linalg.norm(vec)


class MetricProtocol(Protocol):
    @property
    def dim(self) -> int:
        ...

    def __call__(self, x1: np.ndarray, x2: np.ndarray) -> float:
        ...


@dataclass
class AnisotropicMetric:
    basis: List[np.ndarray]
    coefs: np.ndarray
    quasi_factors: np.ndarray

    def __post_init__(self):
        assert self.quasi_factors.shape == (self.dim,)
        assert np.all(np.abs(self.quasi_factors) < 1.0)

    @property
    def dim(self) -> int:
        return len(self.coefs)

    def __call__(self, x1: np.ndarray, x2: np.ndarray) -> float:
        sqdist = 0.0
        diff = x2 - x1
        for c, v, qf in zip(self.coefs, self.basis, self.quasi_factors):
            inpro = v.dot(diff)
            quasi_term = (1 - qf) if inpro >= 0.0 else (1 + qf)
            inpro_modified = inpro / quasi_term
            sqdist_partial = (c * inpro_modified) ** 2
            sqdist += sqdist_partial
        return float(sqrt(sqdist))


class SingleAxisMetric(AnisotropicMetric):
    @staticmethod
    def gram_schmidt(vectors: List[np.ndarray]) -> List[np.ndarray]:
        basis: List[np.ndarray] = []
        for vector in vectors:
            for previous_vector in basis:
                vector -= np.dot(vector, previous_vector) * previous_vector
            norm = np.linalg.norm(vector)
            if norm > 0:
                basis.append(vector / norm)
        return basis

    @classmethod
    def create(
        cls,
        major_axis: np.ndarray,
        coef_major: float = 1.0,
        coef_minor=1e-3,
        quasi_factors: Optional[np.ndarray] = None,
    ) -> AnisotropicMetric:
        ndim = len(major_axis)

        if quasi_factors is None:
            quasi_factors = np.zeros(ndim)

        e1 = normalize(major_axis)
        basis_cand = [e1]
        for _ in range(ndim - 1):
            basis_cand.append(np.random.randn(ndim))
        basis = cls.gram_schmidt(basis_cand)
        coefs = np.array([coef_major] + [coef_minor] * (ndim - 1))
        return cls(basis, coefs, quasi_factors)


class StandardMetric(AnisotropicMetric):
    @classmethod
    def create(
        cls,
        dim: int,
        coefs: Optional[np.ndarray] = None,
        quasi_factors: Optional[np.ndarray] = None,
    ) -> AnisotropicMetric:

        if quasi_factors is None:
            quasi_factors = np.zeros(dim)

        basis = []
        for i in range(dim):
            e = np.zeros(dim)
            e[i] = 1.0
            basis.append(e)
        if coefs is None:
            coefs = np.ones(dim)
        return cls(basis, coefs, quasi_factors)


@dataclass
class RadialSphericalMetric:
    center: np.ndarray
    quasi_factor: float = 0.0

    def __post_init__(self):
        assert np.abs(self.quasi_factor) < 1.0

    @property
    def dim(self) -> int:
        return len(self.center)

    def __call__(self, x1: np.ndarray, x2: np.ndarray) -> float:
        d1 = np.linalg.norm(x1 - self.center)
        d2 = np.linalg.norm(x2 - self.center)
        pure_dist = np.abs(d1 - d2)
        if d1 <= d2:
            quasi_dist = pure_dist / (1 - self.quasi_factor)
        else:
            quasi_dist = pure_dist / (1 + self.quasi_factor)
        return quasi_dist


@dataclass
class AngularSphericalMetric:
    center: np.ndarray

    @property
    def dim(self) -> int:
        return len(self.center)

    def __call__(self, x1: np.ndarray, x2: np.ndarray) -> float:
        d1 = np.linalg.norm(x1 - self.center)
        d2 = np.linalg.norm(x2 - self.center)
        inpro = (x1 - self.center).dot(x2 - self.center)
        pure_dist = acos(inpro / (d1 * d2))
        return pure_dist


@dataclass
class CartesianMetric:
    # cartesian metric space
    metrics: List[MetricProtocol]

    @property
    def dim(self) -> int:
        return sum([m.dim for m in self.metrics])

    def __call__(self, x1: np.ndarray, x2: np.ndarray) -> float:
        dim_list = [m.dim for m in self.metrics]
        x1_split = np.split(x1, np.cumsum(dim_list)[:-1])
        x2_split = np.split(x2, np.cumsum(dim_list)[:-1])

        squared_dists = 0.0
        for metric, x1sp, x2sp in zip(self.metrics, x1_split, x2_split):
            squared_dists += metric(x1sp, x2sp) ** 2
        return np.sqrt(squared_dists)
