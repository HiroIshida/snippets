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


