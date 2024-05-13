## note
- kavraki group icra 2024
- compute stochastic sdf per each link of the robot.
- the stochasticity of the sdf is represented by a mean and a variance
- By doing this, converting the stochastic problem into a deterministic problem.

## question
- I can imagine compute the collision risk for a single configuration, but how can you evaluate it for a motion? It seems that the risk would be dependent on the descritization.
- why does gradient appears?
- I think to simulate sensor, we need to sample the environment. If so, I'd like to know how to sample the environment and if not, I'd like to know how to simulate the sensor. Is there any domain-gap appearing in the learning phase and deployment phase?
- Is there any chance that vectorized motion planning could be applied to this method? And if not so, what is the reason?
- The soft constraints is enforced to guide the IK optimization. How is it usually trapped in the local minimum? And in that case, do you employ random restarts? 
- In my experiences, in an application of model based planning to real robotics, uncertainty due to kinematics / camera calibration error is severe rather than the one in the sensor noise. Is there any implication of this method to the kinematics / camera calibration error?
