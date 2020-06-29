### TODO 
### failure 
cycle1: root = true => enter => result1 = false => break
cycle2: root = false => break

### success
cycle1: root = true => enter => result1 = true  



### code
result1 is evaluated inside following function from `btDbvtBroadphase.h` and `btDbvt.h`.
```cpp
void btDbvtBroadphase::rayTest(const btVector3& rayFrom, const btVector3& rayTo, btBroadphaseRayCallback& rayCallback, const btVector3& aabbMin, const btVector3& aabbMax)
{
	BroadphaseRayTester callback(rayCallback);
	btAlignedObjectArray<const btDbvtNode*>* stack = &m_rayTestStacks[0];
#if BT_THREADSAFE
	// for this function to be threadsafe, each thread must have a separate copy
	// of this stack.  This could be thread-local static to avoid dynamic allocations,
	// instead of just a local.
	int threadIndex = btGetCurrentThreadIndex();
	btAlignedObjectArray<const btDbvtNode*> localStack;
	//todo(erwincoumans, "why do we get tsan issue here?")
	if (0)//threadIndex < m_rayTestStacks.size())
	//if (threadIndex < m_rayTestStacks.size())
	{
		// use per-thread preallocated stack if possible to avoid dynamic allocations
		stack = &m_rayTestStacks[threadIndex];
	}
	else
	{
		stack = &localStack;
	}
#endif

	m_sets[0].rayTestInternal(m_sets[0].m_root,
							  rayFrom,
							  rayTo,
							  rayCallback.m_rayDirectionInverse,
							  rayCallback.m_signs,
							  rayCallback.m_lambda_max,
							  aabbMin,
							  aabbMax,
							  *stack,
							  callback);

	m_sets[1].rayTestInternal(m_sets[1].m_root,
							  rayFrom,
							  rayTo,
							  rayCallback.m_rayDirectionInverse,
							  rayCallback.m_signs,
							  rayCallback.m_lambda_max,
							  aabbMin,
							  aabbMax,
							  *stack,
							  callback);
}
```
```cpp
DBVT_PREFIX
inline void btDbvt::rayTestInternal(const btDbvtNode* root,
									const btVector3& rayFrom,
									const btVector3& rayTo,
									const btVector3& rayDirectionInverse,
									unsigned int signs[3],
									btScalar lambda_max,
									const btVector3& aabbMin,
									const btVector3& aabbMax,
									btAlignedObjectArray<const btDbvtNode*>& stack,
									DBVT_IPOLICY) const
{
	(void)rayTo;
	DBVT_CHECKTYPE
	if (root)
	{
		btVector3 resultNormal;

		int depth = 1;
		int treshold = DOUBLE_STACKSIZE - 2;
		stack.resize(DOUBLE_STACKSIZE);
		stack[0] = root;
		btVector3 bounds[2];
		do
		{
			const btDbvtNode* node = stack[--depth];
			bounds[0] = node->volume.Mins() - aabbMax;
			bounds[1] = node->volume.Maxs() - aabbMin;
			btScalar tmin = 1.f, lambda_min = 0.f;
			unsigned int result1 = false;
			result1 = btRayAabb2(rayFrom, rayDirectionInverse, signs, bounds, tmin, lambda_min, lambda_max);
			if (result1)
			{
				if (node->isinternal())
				{
					if (depth > treshold)
					{
						stack.resize(stack.size() * 2);
						treshold = stack.size() - 2;
					}
					stack[depth++] = node->childs[0];
					stack[depth++] = node->childs[1];
				}
				else
				{
					policy.Process(node);
				}
			}
		} while (depth);
	}
}
```

