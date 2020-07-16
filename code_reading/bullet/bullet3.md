# pybullet$B$N$7$/$_(B
$B%/%i%$%"%s%H$O(B: $B$^$:(Bpybullet.c$BFb$G(B, $B%3%^%s%I$r@8@.$9$k(B(b3CalculateInverseKinematicsCommandInit). . 

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
$B$=$l$>$l$N%3%^%s%I$O(B, $B%3%^%s%I<1JLMQ$N(B`m_type`$B$H$$$&%a%s%P$K2C$($F(B, $B$=$NB>$N=hM}It$K$o$?$9$?$a$N0z?t$r$^$H$a$?9=B$BN(B($BNc$($P(B: $B$rFbIt$K$b$C$F$$$k(B. $B$=$NDj5A$O(B, $B$3$s$J46$8(B. command init$B4X?tFb$G$O>e5-$N(BhogehogeArgs$B$N%a%s%PJQ?t$rKd$a$F$$$/A`:n$r$7$F$$$k$3$H$KCm0U(B.  
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
$B$3$&$7$F$D$/$i$l$?%3%^%s%I$r(B`b3SubmitClientCommandAndWaitStatus`$B$K$h$C$F%5!<%P$KAw$j$D$1$k(B. $B%5!<%P(B`PhysicsServerCommandProcessor.cpp`$BFb$N(Bswitch$BJ8$K$h$C$F(B, dispatch$B$5$l$F:G=*E*$K%a%$%s=hM}4X?t$KE~C#$9$k(B. $B%a%$%s=hM}4X?t$O$<$s$V<!$N7A<0(B:  
```c++
bool PhysicsServerCommandProcessor::processCalculateInverseKinematicsCommand(const struct SharedMemoryCommand& clientCmd, struct SharedMemoryStatus& serverStatusOut, char* bufferServerToClient, int bufferSizeInBytes)
```
$BBg;v$J$N$O(B, `clientCmd`$B$H(B`serverStatusOut`$B$G$"$k(B. $BA0<T$O@h=R$7$?(B, $B%/%i%$%"%s%H$+$iFO$$$?%3%^%s%I$G$"$k(B. $B8e<T$O(B, $B=hM}$K@.8y$7$?$+$I$&$+$N>pJs$H<B9T7k2L%G!<%?$rC_$($?$b$N$G$"$k(B. $B@.8y$N2DH]$O$3$s$J46$8$G%;%C%H$5$l$k(B: 
```
serverCmd.m_type = CMD_CALCULATE_INVERSE_KINEMATICS_COMPLETED;
serverCmd.m_type = CMD_CALCULATE_INVERSE_KINEMATICS_FAILED; 
```
$BJ*M}%(%s%8%s$r%@%$%l%/%H$K;H$&>l9g$H(BGUI(shared memory?)$B$rMQ$$$k>l9g$=$l$>$l(B, 
```c++
void PhysicsDirect::postProcessStatus(const struct SharedMemoryStatus& serverCmd)
const SharedMemoryStatus* PhysicsClientSharedMemory::processServerStatus()
```
$B$N$h$&$J46$8$G%A%'%C%/$5$l$k$h$&$J$N$G(B, $B?7$7$$%a%=%C%I$rDI2C$9$k>l9g$K$ON>J}$K(Bcase$B$rDI2C$9$k$3$H$r$*K:$l$J$/(B. $B$3$N%A%'%C%/$,=*$o$C$?$"$H$O(B, `b3SubmitClientCommandAndWaitStatus`$B$N7k2L$H$7$F(Bstatus$B$N%O%s%I%k$,JV$C$F$/$k$N$G(B, $B$3$NFbItJQ?t$r<h$j=P$9$3$H$G(B, $B=jK>$N%G!<%?$rF@$k$3$H$,$G$-$k(B. client$B$N;~$HF1$8$h$&$K%5!<%P$,JV$7$F$/$k(B(sever)status$B$bFbIt$K%G!<%?J]B8MQ9=B$BN$r$b$C$F$$$k(B. $B$?$H$($P(B: 
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
$B$N$h$&$J46$8(B. $B:G8e$K(B, `CMD_HOGE_HOGE`$B$N$h$&$J$b$N$O$9$Y$F(B`SharedMemoryPubulic.h`$BFb$G(Benum$B$GDj5A$5$l$F$$$k$3$H$KCm0U(B.  



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

