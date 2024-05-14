Discontinuity-Sensitive Optimal Control Learning by Mixture of Experts, Gan Tang, Kris Hauser, ICRA 2019

## big idea
- In the problem space for the optimal control policy becomes discontinuous, nearest neighbor does not work well.
- So partitioning the P-space into regions where the optimal control policy is continuous is a good idea.

## method
### ahead of time
- sample problem and solve it using the optimal control policy then create a database
- Apply a clustering algorithm to the database, where the optimal trajectory rather than the problem vector is used as the feature vector.
- Train individual models for each cluster that predicts optimal control input given the problem vector.
- clustering results are also to use to train an gating network that selects the appropriate model for a given problem vector.
### online
- the prediction of the regressors-classifier-joint model predicts the optimal control input from the problem vector.
- LQR is used to track the predicted trajectory.
- After it reaches the end of the trajectory, then switches to stabilization mode to reach the goal.
- Note that predicted trajectory's terminal state is the goal state, thus the stabilization mode is not necessary.
