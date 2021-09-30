#!/usr/bin/env python
from numpy.random import *
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import re
import inspect
from scipy.interpolate import RegularGridInterpolator

class SignedDistanceField:

    def __init__(self, sdffile, scale = 1.0):

        with open(sdffile) as f:
            lines = f.readlines()

        Nxyz = np.array(map(lambda str: int(str), re.sub('\n', '', lines[0]).split()))
        bmin = np.array(map(lambda str: float(str), re.sub('\n', '', lines[1]).split())) * scale

        dx = float(re.sub('\n', '', lines[2])) * scale

        data_raw = map(lambda str: float(re.sub('\n', '', str)) * scale, lines[3:]) 
        Nx = Nxyz[0]
        Ny = Nxyz[1]
        Nz = Nxyz[2]

        data_ = np.zeros([Nx, Ny, Nz], dtype=float)
        for k in range(Nz):
            for j in range(Ny):
                for i in range(Nx):
                    idx = i + j*Nx + k*Nx*Ny
                    data_[i, j, k] = data_raw[idx]

        data = np.array(data_)
        xlin = [bmin[0]+dx*i for i in range(Nx)]
        ylin = [bmin[1]+dx*i for i in range(Ny)]
        zlin = [bmin[2]+dx*i for i in range(Nz)]
        itp = RegularGridInterpolator((xlin, ylin, zlin), data)

        self.data = data
        self.bmin = bmin
        self.bmax = bmin + (Nxyz-1)*dx
        self.Nxyz = Nxyz
        self.dx = dx
        self.itp = itp

    def sd(self, p):
        if self.isInsideDefBox(p):
            return self.itp(p).item()
        else:
            return float('inf')

    """
    def grad(self, p, eps = 0.00000001):
        # the reason why using randn to make basis is to avoid singularity
        def normalize(vec):
            n = np.linalg.norm(vec)
            return vec/n
        ex = normalize(randn(3))
        ey = normalize(np.cross(ex, randn(3)))
        ez = normalize(np.cross(ex, ey))
        f0 = self.sd(p)
        fxp = self.sd(p + ex * eps)
        fyp = self.sd(p + ey * eps)
        fzp = self.sd(p + ez * eps)
        grad_tmp = np.array([(fxp - f0)/eps, (fyp - f0)/eps, (fzp - f0)/eps])
        mat = np.vstack([ex, ey, ez])
        grad = mat.dot(grad_tmp)
        return grad
        """

    def show(self, z):
        Nxyz = self.Nxyz
        Nx = Nxyz[0]
        Ny = Nxyz[1]
        Nz = Nxyz[2]
        xlin = [self.bmin[0]+self.dx*i for i in range(Nx)]
        ylin = [self.bmin[1]+self.dx*i for i in range(Ny)]
        zlin = [self.bmin[2]+self.dx*i for i in range(Nz)]
        A, B = np.meshgrid(ylin, xlin) 
        Z = self.data[:, :, z]
        plt.pcolor(A, B, Z.T)
        plt.colorbar()
        plt.show()

    def show_ptcls(self):
        N = 100000
        x_arr = rand(N) * (self.bmax[0] - self.bmin[0]) + self.bmin[0]
        y_arr = rand(N) * (self.bmax[1] - self.bmin[1]) + self.bmin[1]
        z_arr = rand(N) * (self.bmax[2] - self.bmin[2]) + self.bmin[2]

        idx_lst = []
        for i in range(N):
            x = x_arr[i]
            y = y_arr[i]
            z = z_arr[i]
            if self.sd([x, y, z]) < 2:
                idx_lst.append(i)
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(x_arr[idx_lst], y_arr[idx_lst], z_arr[idx_lst])
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("angle [rad]")
        plt.show()

    def isInsideDefBox(self, p):
        for i in range(3):
            if p[i] < self.bmin[i]:
                return False
            if p[i] > self.bmax[i]:
                return False
        return True

if __name__=='__main__':
    sdf = SignedDistanceField("./stanford.sdf")
    sdf.show(20)


