import numpy as np

class KalmanFilter: # not really a kalman filter...
    def __init__(self):
        self.x_est = None
        self.P = None
        self.isInitialized = False

    def initialize(self, x_mean, x_cov):
        self.x_est = x_mean
        self.P = x_cov
        self.isInitialized = True

    def update(self, z_obs, R):
        # notation is similar to wikipedia
        y = z_obs - self.x_est # H = eye
        S = self.P + R
        K = self.P.dot(np.linalg.inv(S))
        x_est_new = self.x_est + K.dot(y)
        P_new = (np.eye(3) - K).dot(self.P)
        self.x_est = x_est_new
        self.P = P_new

    def get_current_est(self, withCov=False):
        if withCov:
            cov = self.P
        else:
            cov = None
        return self.x_est, cov
