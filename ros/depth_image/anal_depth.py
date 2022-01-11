import numpy as np
import pickle
import scipy
import cv_bridge
import matplotlib.pyplot as plt
import scipy.interpolate

with open('depth.pkl3', 'rb') as f:
    depth = pickle.load(f)

buf = np.ndarray(shape=(1, int(len(depth.data)/4)),
                  dtype=np.float32, buffer=depth.data)

depth = buf.reshape([480, 640])

def create_mesh_points(width, height):
    xlin = np.linspace(0, 1., width)
    ylin = np.linspace(0, 1., height)
    xx, yy = np.meshgrid(xlin, ylin)
    pts = np.array(list(zip(xx.flatten(), yy.flatten())))
    return pts

pts_in = create_mesh_points(*depth.shape)
pts_out = create_mesh_points(200, 200)

itped = scipy.interpolate.griddata(pts_in, depth.T.flatten(), pts_out)
itped_reshaped = itped.reshape(200, 200).T

plt.imshow(itped_reshaped)
plt.show()
