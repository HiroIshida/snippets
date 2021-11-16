以下のようなコードだと: 
```c++
const lanelet::CompoundPolygon3d llt_poly = llt.polygon3d();
const auto lb = llt.leftBound2d();
const auto rb = llt.rightBound2d();

RCLCPP_INFO_STREAM(rclcpp::get_logger("ishida_debug"), "left_bound: " << lb);
RCLCPP_INFO_STREAM(rclcpp::get_logger("ishida_debug"), "rigth_bound: " << rb);
for(const auto& p : llt_poly){
RCLCPP_INFO_STREAM(rclcpp::get_logger("ishida_debug"), "point: " << p);
}
```

次のようなアウトプットがでて, 
```
1637076315.2237506 [parking_route_planner_debug_node-17] [INFO] [1637076315.216321882] [ishida_debug]: left_bound: [id: 486 point ids: 229, 230, 232, 233]
1637076315.2237945 [parking_route_planner_debug_node-17] [INFO] [1637076315.216330034] [ishida_debug]: rigth_bound: [id: 488 point ids: 221, 222, 224, 225]
1637076315.2238381 [parking_route_planner_debug_node-17] [INFO] [1637076315.216360987] [ishida_debug]: point: [id: 229 x: 86127.8 y: 43000.7 z: -1.7333]
1637076315.2238815 [parking_route_planner_debug_node-17] [INFO] [1637076315.216373129] [ishida_debug]: point: [id: 230 x: 86127.5 y: 43001.2 z: -1.7042]
1637076315.2239246 [parking_route_planner_debug_node-17] [INFO] [1637076315.216380673] [ishida_debug]: point: [id: 232 x: 86127.1 y: 43001.8 z: -1.6901]
1637076315.2239680 [parking_route_planner_debug_node-17] [INFO] [1637076315.216387836] [ishida_debug]: point: [id: 233 x: 86126.5 y: 43002.7 z: -1.6771]

1637076315.2240112 [parking_route_planner_debug_node-17] [INFO] [1637076315.216396019] [ishida_debug]: point: [id: 225 x: 86128.1 y: 43003.3 z: -1.6662]
1637076315.2240543 [parking_route_planner_debug_node-17] [INFO] [1637076315.216403540] [ishida_debug]: point: [id: 224 x: 86128.5 y: 43002.6 z: -1.6973]
1637076315.2241023 [parking_route_planner_debug_node-17] [INFO] [1637076315.216410996] [ishida_debug]: point: [id: 222 x: 86129 y: 43001.9 z: -1.7195]
1637076315.2241466 [parking_route_planner_debug_node-17] [INFO] [1637076315.216418393] [ishida_debug]: point: [id: 221 x: 86129.4 y: 43001.6 z: -1.743]
```

それをpythonでplotすると, どうやら, leftbound, rightboundは側面のことのようだ. 
```python
import matplotlib.pyplot as plt
import numpy as np

X = np.array([
    [86127.8, 43000.7], 
    [86127.5, 43001.2], 
    [86127.1, 43001.8], 
    [86126.5, 43002.7],
    [86128.1, 43003.3],
    [86128.5, 43002.6],
    [86129, 43001.9],
    [86129.4, 43001.6]])

plt.scatter(X[:, 0], X[:, 1])
plt.axis('scaled')
plt.show()
```
