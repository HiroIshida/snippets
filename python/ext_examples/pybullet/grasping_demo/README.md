## Get started
- First, you should download PR2 mesh files from using [gdown](https://github.com/wkentaro/gdown).
- On grasping_demo folder layer, execute below. You can get the meshes folder in the pr2_description folder.
```
pip install gdown
python scripts/script.py
```


## models
The dish model is downloaded from [here](https://creazilla.com/nodes/71885-candy-dish-3d-model), then number of vertices is reduced and rescaled (x0.001) using meshlab.
The gripper model is Fetch's gripper, urdf of which is modified so that it's base has free 3dof planar joint.
