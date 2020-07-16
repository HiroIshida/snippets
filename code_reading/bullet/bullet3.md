# implemenation of pybullet
First, inside `pyblullet.c`, client side generate a command by HogeHogeCommandInit (like b3CalculateInverseKinematicsCommandInit).
```c++
// inside PhysicsClientC_API.cpp
B3_SHARED_API b3SharedMemoryCommandHandle b3CalculateInverseKinematicsCommandInit(b3PhysicsClientHandle physClient, int bodyUniqueId){
	PhysicsClient* cl = (PhysicsClient*)physClient;
	struct SharedMemoryCommand* command = cl->getAvailableSharedMemoryCommand();
	command->m_type = CMD_CALCULATE_INVERSE_KINEMATICS;
	command->m_calculateInverseKinematicsArguments.m_bodyUniqueId = bodyUniqueId;
	return (b3SharedMemoryCommandHandle)command;
}
```
each command from the client side, has a member `m_type=CMD_HOGEHOGE` which later will be used to identify the command. Also, the command has a struct which equipped with all the arguments that will be passe d to main procedure. Note that inside `hogehogeCommandInit` these member variables of the struct is filled as follows:
```c++
// inside SharedMemoryCommands.h
struct CalculateInverseKinematicsResultArgs{
	int m_bodyUniqueId;
	int m_dofCount;
	double m_jointPositions[MAX_DEGREE_OF_FREEDOM];
};
// blah blah 
struct CalculateInverseKinematicsArgs m_calculateInverseKinematicsArguments;
```
Then this command will be sent to the server side by `b3SubmitClientCommandAndWaitStatus`. In server side `PhysicsServerCommandProcessor.cpp`, through the switch sequences, the command is dispatched to the main procedure. All such main procedure is of the form:
```c++
bool PhysicsServerCommandProcessor::processCalculateInverseKinematicsCommand(const struct SharedMemoryCommand& clientCmd, struct SharedMemoryStatus& serverStatusOut, char* bufferServerToClient, int bufferSizeInBytes)
```
Here important arguments are `clientCmd`and`serverStatusOut`, the former of which are explained already. The later has 1) information of success or not and 2) data obtained by execution. For example of inverse kinematics case, `serverStatusOut` will contain success and solved joint angles. Information of success/not is set as follows:
```
serverCmd.m_type = CMD_CALCULATE_INVERSE_KINEMATICS_COMPLETED;
serverCmd.m_type = CMD_CALCULATE_INVERSE_KINEMATICS_FAILED; 
```
Note that using this success/not info, finally client side check the status. This check process is different for DIRECT and GUI (shared?) case:
```c++
void PhysicsDirect::postProcessStatus(const struct SharedMemoryStatus& serverCmd)
const SharedMemoryStatus* PhysicsClientSharedMemory::processServerStatus()
```
Thus, if you wanna add a new function, then don't forget add checks for both cases. After this checking process, the execution result is return by `b3SubmitClientCommandAndWaitStatus` (finally!), which then will be returned from `pybullet.c`. LIke client side, the server side status class also has specific struct to save returning data :
```
// inside SharedMemoryCommands.h
struct CalculateInverseKinematicsResultArgs
{
	int m_bodyUniqueId;
	int m_dofCount;
	double m_jointPositions[MAX_DEGREE_OF_FREEDOM];
};
// blah blah 
struct CalculateInverseKinematicsResultArgs m_inverseKinematicsResultArgs;
```
Note that all `CMD_HOGE_HOGE` kind things are defined in `SharedMemoryPubulic.h` using enum.

# pitfalls
## when joint limit is not specified in urdf
lower and upper limits are set to 0.0 and -1.0. As the bullet (not pybullet) documentation shows, if lower > upper, that means the joint is free.

## calculateJacobian
Please specify that the base is fixed:
```python
robot = pb.loadURDF("./robot/config.urdf", useFixedBase=True)
```
otherwise the `calcualateJacobian` will return 3x(n+6) matrix. The additional 6 dims are for xyzrpy.

## getCameraImage
`bullet3/examples/SharedMemory/plugins/tinyRendererPlugin`
`bullet3/examples/OpenGLWindow`

## about rayTest
Declaration:
```cpp
virtual void rayTest(const btVector3& rayFromWorld, const btVector3& rayToWorld, RayResultCallback& resultCallback) const;
```
First by using a aabb collision checking (collision of aligned boxes) which implemented in `BroadPhaseCollision`, then perform precise collision checking by `rayTestSingleInternal`, where different processes are used for convex and non-convex object cases.
```cpp
void btCollisionWorld::rayTestSingleInternal(const btTransform& rayFromTrans, const btTransform& rayToTrans,
											 const btCollisionObjectWrapper* collisionObjectWrap,
											 RayResultCallback& resultCallback)
```

## how inverse kinematics method is imported into python
1. `BussIK` defined main method. `Jacobian::CalcDeltaThetas*` compute `dq` and set it to a public property `dTheta`.
2. `IKTrajectoryHelper.h` wraps `BussIK`. `computeIK` method call `CalcDeltaThetas*` mentioned above and set `q_new = q_current + dq` (`*q_new` is returned as a argument).
3. `PhysicsServerCommandProcessor` calls `computeIK` inside `processCalculateInverseKinematicsCommand`.

defined in `BussIkjk` then wrapped by `IKTrajectoryHelper.h`. This header is included by `PhysicsServerCommandProcessor.cpp`. Then finally integrated into `*C_API`.

## calcualateJacobians 
Super confusing thing is, `calcualateJacobians` method seem to come from `src/BulletInverseDynamics/MultBodyTree.hpp`. Insider `PhysicsServerCommandProcessor::processCalculateJacobianCommand`,  `tree->getBodyJacobianTrans` and `tree->getBodyJacobianRot` are called and computed values are stored as `jac_t` and `jac_r`. Seems that `jac_r` is used just to compute `jac_t_new`, and not returned.

