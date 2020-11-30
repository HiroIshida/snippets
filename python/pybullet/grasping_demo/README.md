## Get started
- First, you should download PR2 mesh files from using [gdown](https://github.com/wkentaro/gdown).
```
pip install gdown
gdown https://drive.google.com/uc?id=1OXyxBEqamCg7cVnLmLj8WvPWRKFEldsC
```
- Put mesh folder under pr2_description.

## models
The dish model is downloaded from [here](https://creazilla.com/nodes/71885-candy-dish-3d-model), then number of vertices is reduced and rescaled (x0.001) using meshlab.
The gripper model is Fetch's gripper, urdf of which is modified so that it's base has free 3dof planar joint.

