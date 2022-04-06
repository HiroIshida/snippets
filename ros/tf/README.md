## ROS (ROS2) のTFの定義

```
tf_source_to_target = self.listener.lookupTransform(
    target_frame, source_frame, rospy.Time(0))
```
としたとき, source = camera, target=baseとしたとき, z=1.7となるので. targetからみたsourceの座標系となっている. つまりsourceの基底をtargetの基底に変換している. 
cppの場合同様の方法で取得した２つの`tf_a2b` と `tf_b2c` は `tf_a2c = tf_a2b * tf_b2c` になる. 
cpp_tf_example/を参照

## 
```python
import rospy
import tf

rospy.init_node('hoge')
target_frame  = "/torso_lift_link"
source_frame = "/base_footprint"
listener = tf.TransformListener()
listener.waitForTransform(target_frame, source_frame, rospy.Time(), rospy.Duration(4.0))
tf = listener.lookupTransform(target_frame, source_frame, rospy.Time(0))
print(tf)
"""return
([0.05, 0.0, -0.8588468872983683], [-0.0, -0.0, -0.0, 1.0])
"""
```

