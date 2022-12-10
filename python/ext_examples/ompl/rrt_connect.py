import matplotlib.pyplot as plt
import numpy as np
from ompl.geometric import SimpleSetup, RRTConnect
from ompl.base import RealVectorStateSpace, State, StateValidityCheckerFn

space = RealVectorStateSpace()
# everytime when we call addDimension, the dimension of the space is increased
space.addDimension(0.0, 1.0)
space.addDimension(0.0, 1.0)

simple_setup = SimpleSetup(space)
algorithm = RRTConnect(simple_setup.getSpaceInformation())
algorithm.setRange(0.05)
simple_setup.setPlanner(algorithm)

def is_valid(state: State):
    r = np.linalg.norm(np.array([state[0], state[1]]) - np.array([0.5, 0.5]))
    return r > 0.35

simple_setup.setStateValidityChecker(StateValidityCheckerFn(is_valid))

start = State(simple_setup.getStateSpace())
start()[0] = 0.1
start()[1] = 0.1

goal = State(simple_setup.getStateSpace())
goal()[0] = 0.9
goal()[1] = 0.9

simple_setup.setStartAndGoalStates(start, goal)

simple_setup.solve()
simple_setup.getSolutionPath()

states = simple_setup.getSolutionPath().getStates()

points = [np.array([s[0], s[1]]) for s in states]

for i in range(len(points) - 1):
    p0 = points[i]
    p1 = points[i + 1]
    plt.plot([p0[0], p1[0]], [p0[1], p1[1]])
plt.show()
