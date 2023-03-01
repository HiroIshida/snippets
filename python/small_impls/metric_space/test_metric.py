from math import sqrt

import numpy as np

from quotdim.metric import (
    AngularSphericalMetric,
    CartesianMetric,
    RadialSphericalMetric,
    SingleAxisMetric,
    StandardMetric,
)


def assert_almost_equal(a: float, b: float, eps=1e-6) -> None:
    absdiff = abs(a - b)
    assert absdiff < eps


def test_anisotropic_metric():
    # single axis
    metric = SingleAxisMetric.create(np.ones(3), coef_minor=0.0)
    assert_almost_equal(metric(np.zeros(3), np.ones(3)), np.sqrt(3))

    metric = SingleAxisMetric.create(np.array([1, 0, 0]), coef_minor=0.0)
    assert_almost_equal(metric(np.zeros(3), np.ones(3)), 1)
    assert_almost_equal(metric(np.zeros(3), np.array([0, 1, 1])), 0.0)

    # single axis with quasi factor
    metric = SingleAxisMetric.create(
        np.array([1, 0, 0]), coef_minor=0.0, quasi_factors=np.array([0.5, 0.0, 0.0])
    )
    assert_almost_equal(metric(np.zeros(3), np.ones(3)), 2.0)
    assert_almost_equal(metric(np.zeros(3), -np.ones(3)), 1.0 / 1.5)

    # standard metric
    metric = StandardMetric.create(3, np.array([1, 2, 3]))
    assert_almost_equal(metric(np.zeros(3), np.ones(3)), sqrt(14))


def test_radial_metric():
    metric = RadialSphericalMetric(np.zeros(3), quasi_factor=0.5)
    assert_almost_equal(metric(np.zeros(3), np.ones(3)), np.sqrt(3) / 0.5)
    assert_almost_equal(metric(np.ones(3), np.zeros(3)), np.sqrt(3) / 1.5)


def test_angular_metric():
    metric = AngularSphericalMetric(np.zeros(3))
    assert_almost_equal(metric(np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0])), np.pi * 0.5)


def test_cartesian_metric():
    metric1 = AngularSphericalMetric(np.zeros(3))
    metric2 = StandardMetric.create(3)
    metric = CartesianMetric([metric1, metric2])
    assert metric.dim == 6

    x1 = np.array([1, 0, 0, 0, 0, 0])
    x2 = np.array([0, 1, 0, 1, 1, 1])
    assert_almost_equal(metric(x1, x2), np.sqrt(0.25 * np.pi**2 + 3))
