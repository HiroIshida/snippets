## pitfalls in conversion from OccupancyGrid msg to costmap function
1. usually we only care about costmap wrt base link. Thus we need to compute `tf_base_to_map`. For this, two conversions are required:
```python
tf_base_to_map = tf_base_to_odom * tf_odom_to_map
```
where `tf_odom_to_map` is given in `msg.info.origin. Note that `OccupancyGrid` msg is (usually) defined in `odom_combined` frame (but check `msg.header.frame_id`).

2. after you numpy-nize the array like `np.array(msg.data).reshape((N, N))` you must transpose it!
```python
info = msg.info
n_grid = np.array([info.width, info.height])
tmp = np.array(msg.data).reshape((n_grid[1], n_grid[0])) # about to be transposed!!
arr = tmp.T # [IMPORTANT] !!
```

