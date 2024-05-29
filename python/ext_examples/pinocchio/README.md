## pitfalls
1. joint name can be found in `model.names`
2. unbounded rotational joint has dimension 2
3. need to specify `pin.ReferenceFrame.LOCAL_WORLD_ALIGNED` to get a jacobian in natural (?) sense
4. need to call `model.computeJointJacobians(av)` before calling `model.getFrameJacobian`.

## add frame to the robot
Frameクラスの引数はこんな感じ. parentとpreviousFrameどちらを使えばよいのか.
```
FrameTpl(const std::string & name,
         const JointIndex parent,
         const FrameIndex previousFrame,
         const SE3 & frame_placement,
         const FrameType type,
         const Inertia & inertia = Inertia::Zero())
```
addFrameの実装を読んで判断していく. 

## tips
add new link. parent frameも指定するようになっているが, parent jointを指定すれば一意に定まるはずなのでなんで必要なのかわからない. 今回はとりあえず0にしてる.
frame とそのparent jointの座標は一致していることに注意.

```python
placement = pin.SE3.Identity()
placement.translation = np.array([0.1, 0.1, 0.1])
new_frame = pin.Frame("new_link", attach_joint_id, 0, placement, pin.FrameType.OP_FRAME)
model.model.addFrame(new_frame, append_inertia=False)
```
