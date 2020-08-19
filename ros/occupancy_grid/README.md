## pitfalls in conversion from OccupancyGrid msg to costmap function
0. ros's `tf.transformations` is originated from [Gohlke's transformations](https://pypi.org/project/transformations/). but seems to use a older version (version of 2009). On the other hand for example in [meshcat-python](https://github.com/rdeits/meshcat-python) of Deits uses 2015 version. The confusing thing is two version use different definition of quaternion. Ver 2009 uses the order of `[x, y, z, w]` but, Ver 2015 uses `[w, x, y, z]`. うんこ


1. usually we only care about costmap wrt base link. Thus we need to compute `tf_base_to_map`. For this, two conversions are required:
```python
tf_base_to_map = tf_base_to_odom * tf_odom_to_map
```
where `tf_odom_to_map` is given in `msg.info.origin`. Note that `OccupancyGrid` msg is (usually) defined in `odom_combined` frame (but check `msg.header.frame_id`).

2. after you numpy-nize the array like `np.array(msg.data).reshape((N, N))` you must transpose it!
```python
info = msg.info
n_grid = np.array([info.width, info.height])
tmp = np.array(msg.data).reshape((n_grid[1], n_grid[0])) # about to be transposed!!
arr = tmp.T # [IMPORTANT] !!
```

