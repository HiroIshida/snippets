from typing import Callable, Optional, Tuple

import numpy as np

# the following code is ported from my old julia code
# https://github.com/HiroIshida/julia_motion_planning/blob/master/src/double_integrator.jl


def bisection_newton(
    f: Callable[[float], float],
    df: Callable[[float], float],
    start: float,
    end: float,
    tol: float = 0.05,
    max_iter: int = 20,
) -> float:
    est = (start + end) * 0.5
    for _ in range(max_iter):
        fratio = f(est) / df(est)

        if (end - (est - fratio)) * ((est - fratio) - start) < 0.0 or abs(fratio) < (
            end - start
        ) / 4.0:
            if f(est) > 0:
                end = est
            else:
                start = est
            fratio = est - (end + start) * 0.5
        est -= fratio

        if abs(fratio) < tol:
            break
    return est


def optimal_cost(s0: np.ndarray, s1: np.ndarray) -> Tuple[float, float]:
    x0 = s0[:2]
    v0 = s0[2:4]
    x1 = s1[:2]
    v1 = s1[2:4]
    x_diff = x1 - x0
    v_diff = v1 - v0

    p = -4 * (np.dot(v0, v0) + np.dot(v1, v1) + np.dot(v0, v1))
    q = 24 * np.dot(v0 + v1, x_diff)
    r = -36 * np.dot(x_diff, x_diff)

    def cost(t: float) -> float:
        return (
            t
            + np.dot(v_diff, 4.0 * v_diff / t - 6 * (-v0 * t + x_diff) / t**2)
            + np.dot(-6 * v_diff / t**2 + 12 * (x_diff - v0 * t) / t**3, -v0 * t + x_diff)
        )

    def cost_d1(t: float) -> float:
        return t**4 + p * t**2 + q * t + r

    def cost_d2(t: float) -> float:
        return 4.0 * t**3 + 2 * p * t + q

    t_min = 0.0
    t_max = 10.0
    optimal_t = bisection_newton(cost_d1, cost_d2, t_min, t_max)
    return cost(optimal_t), optimal_t


def forward_reachable_box(x0, v0, r):
    tau_x_plus = 2 / 3 * (-(v0**2) + r + v0 * np.sqrt(v0**2 + r))
    tau_x_minus = 2 / 3 * (-(v0**2) + r - v0 * np.sqrt(v0**2 + r))
    xmax = v0 * tau_x_plus + x0 + np.sqrt(1 / 3 * tau_x_plus**2 * (-tau_x_plus + r))
    xmin = v0 * tau_x_minus + x0 - np.sqrt(1 / 3 * tau_x_minus**2 * (-tau_x_minus + r))

    tau_v_plus = 0.5 * r
    vmax = v0 + np.sqrt(tau_v_plus * (-tau_v_plus + r))
    vmin = v0 - np.sqrt(tau_v_plus * (-tau_v_plus + r))
    return xmin, xmax, vmin, vmax


def backward_reachable_box(x0, v0, r):
    tau_x_plus = 2 / 3 * (v0**2 - r + v0 * np.sqrt(v0**2 + r))
    tau_x_minus = 2 / 3 * (v0**2 - r - v0 * np.sqrt(v0**2 + r))
    xmax = v0 * tau_x_plus + x0 + np.sqrt(1 / 3 * tau_x_plus**2 * (tau_x_plus + r))
    xmin = v0 * tau_x_minus + x0 - np.sqrt(1 / 3 * tau_x_minus**2 * (tau_x_minus + r))

    tau_v_plus = 0.5 * r
    vmax = v0 + np.sqrt(tau_v_plus * (tau_v_plus + r))
    vmin = v0 - np.sqrt(tau_v_plus * (tau_v_plus + r))
    return xmin, xmax, vmin, vmax


def gen_trajectory(s0, s1, tau, N_split=10):
    x0 = s0[:2]
    v0 = s0[2:4]
    x1 = s1[:2]
    v1 = s1[2:4]
    x01 = x1 - x0
    v01 = v1 - v0
    d = np.hstack(
        [
            -6 * v01 / tau**2 + 12 * (-tau * v0 + x01) / tau**3,
            4 * v01 / tau - 6 * (-tau * v0 + x01) / tau**2,
        ]
    )

    def f(t):
        s = t - tau
        eye = np.eye(2)
        M_left = np.vstack([np.hstack([eye, eye * s]), np.hstack([np.zeros((2, 2)), eye])])
        M_right = np.vstack(
            [
                np.hstack([eye * (-(s**3)) / 6.0, eye * s**2 * 0.5]),
                np.hstack([eye * (-(s**2) * 0.5), eye * s]),
            ]
        )
        return M_left @ s1 + M_right @ d

    waypoints = [f(t) for t in np.linspace(0, tau, N_split + 1)]
    return waypoints


class TrajectoryPiece:
    d: np.ndarray
    duration: float
    s1: np.ndarray

    def __init__(self, s0, s1, duration: Optional[float] = None):
        if duration is None:
            _, duration = optimal_cost(s0, s1)
        x0 = s0[:2]
        v0 = s0[2:4]
        x1 = s1[:2]
        v1 = s1[2:4]
        x01 = x1 - x0
        v01 = v1 - v0
        d = np.hstack(
            [
                -6 * v01 / duration**2 + 12 * (-duration * v0 + x01) / duration**3,
                4 * v01 / duration - 6 * (-duration * v0 + x01) / duration**2,
            ]
        )
        self.d = d
        self.duration = duration
        self.s1 = s1

    def interpolate(self, t):
        s = t - self.duration
        eye = np.eye(2)
        M_left = np.vstack([np.hstack([eye, eye * s]), np.hstack([np.zeros((2, 2)), eye])])
        M_right = np.vstack(
            [
                np.hstack([eye * (-(s**3)) / 6.0, eye * s**2 * 0.5]),
                np.hstack([eye * (-(s**2) * 0.5), eye * s]),
            ]
        )
        return M_left @ self.s1 + M_right @ self.d


if __name__ == "__main__":
    s0 = np.array([0.2, 0.3, 0, 0])
    s1 = np.array([1, 1, 0, 0])
    c, t = optimal_cost(s0, s1)
    np.testing.assert_approx_equal(c, 3.367317015231521, significant=4)
    np.testing.assert_approx_equal(t, 2.5260490633667847, significant=4)

    x_min, x_max, v_min, v_max = forward_reachable_box(s0[:2], s0[2:4], 1.0)
    np.testing.assert_almost_equal(
        x_min, np.array([-0.022222222222222227, 0.07777777777777775]), decimal=4
    )
    np.testing.assert_almost_equal(
        x_max, np.array([0.4222222222222223, 0.5222222222222223]), decimal=4
    )
    np.testing.assert_almost_equal(v_min, np.array([-0.5, -0.5]), decimal=4)
    np.testing.assert_almost_equal(v_max, np.array([0.5, 0.5]), decimal=4)

    x_min, x_max, v_min, v_max = backward_reachable_box(s0[:2], s0[2:4], 1.0)
    np.testing.assert_almost_equal(
        x_min, np.array([-0.022222222222222227, 0.07777777777777775]), decimal=4
    )
    np.testing.assert_almost_equal(
        x_max, np.array([0.4222222222222223, 0.5222222222222223]), decimal=4
    )
    np.testing.assert_almost_equal(
        v_min, np.array([-0.8660254037844386, -0.8660254037844386]), decimal=4
    )
    np.testing.assert_almost_equal(
        v_max, np.array([0.8660254037844386, 0.8660254037844386]), decimal=4
    )

    pts = gen_trajectory(s0, s1, 1.0, 4)
    np.testing.assert_almost_equal(pts[0], np.array([0.2, 0.3, 0, 0]), decimal=4)
    np.testing.assert_almost_equal(
        pts[1],
        np.array(
            [0.32499999999999996, 0.40937500000000004, 0.9000000000000004, 0.7874999999999996]
        ),
        decimal=4,
    )
    np.testing.assert_almost_equal(
        pts[2],
        np.array([0.5999999999999999, 0.65, 1.2000000000000002, 1.0499999999999998]),
        decimal=4,
    )
    np.testing.assert_almost_equal(
        pts[3], np.array([0.875, 0.890625, 0.9000000000000001, 0.7874999999999999]), decimal=4
    )
    np.testing.assert_almost_equal(pts[4], np.array([1.0, 1.0, 0.0, 0.0]), decimal=4)
