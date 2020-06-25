## pitfalls
1. Do not set position and quaternion when loading URDF. It's buggy.
2. A base link always has -1 index

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

