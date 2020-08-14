import transformations as tfs
import numpy as np
import numpy.random as rn
from math import *
import autograd
import copy

"""
jacobian drpy/dquat
"""
"""
void getRPY(double &roll,double &pitch,double &yaw) const
{
double sqw;
double sqx;
double sqy;
double sqz;

sqx = this->x * this->x;
sqy = this->y * this->y;
sqz = this->z * this->z;
sqw = this->w * this->w;

// Cases derived from https://orbitalstation.wordpress.com/tag/quaternion/
double sarg = -2 * (this->x*this->z - this->w*this->y);
const double pi_2 = 1.57079632679489661923;
if (sarg <= -0.99999) {
  pitch = -pi_2;
  roll  = 0;
  yaw   = 2 * atan2(this->x, -this->y);
} else if (sarg >= 0.99999) {
  pitch = pi_2;
  roll  = 0;
  yaw   = 2 * atan2(-this->x, this->y);
} else {
  pitch = asin(sarg);
  roll  = atan2(2 * (this->y*this->z + this->w*this->x), sqw - sqx - sqy + sqz);
  yaw   = atan2(2 * (this->x*this->y + this->w*this->z), sqw + sqx - sqy - sqz);
}
"""

def myrpy(q):
    x, y, z, w = q
    xx = x * x
    yy = y * y
    zz = z * z
    ww = w * w

    xzmwy= x*z - w*y
    pitch = asin(-2 * xzmwy);
    roll = atan2(2 * (y*z + w*x), ww - xx - yy + zz);
    yaw = atan2(2 * (x*y +z*w), ww + xx - yy - zz);
    return [roll, pitch, yaw]

def jac_rpy_quat(q):
    x, y, z, w = q
    xx = x * x
    yy = y * y
    zz = z * z
    ww = w * w

    def atan2_grad(y, x): 
        n = x**2 + y**2
        return np.array([[-y/n, x/n]])

    xzmwy= x*z - w*y
    dxzmwy = np.array([z, -w, x, -y])
    dpitch = 1/sqrt(1 - xzmwy**2) * (-2 * dxzmwy)

    tmp1 = 2 * y*z + w*x
    tmp2 = ww - xx - yy + zz
    tmp1_grad = 2 * np.array([w, z, z, x])
    tmp2_grad = np.array([-2*x, -2*y, 2*z, 2*w])
    tmp_grad = np.vstack([tmp1_grad, tmp2_grad])
    droll = atan2_grad(tmp1, tmp2).dot(tmp_grad)

    tmp1 = 2 * x*y + w*z
    tmp2 = ww + xx - yy - zz
    tmp1_grad = 2 * np.array([y, x, w, z])
    tmp2_grad = np.array([2*x, -2*y, -2*z, 2*w])
    tmp_grad = np.vstack([tmp1_grad, tmp2_grad])
    dyaw = atan2_grad(tmp1, tmp2).dot(tmp_grad)

    J = np.vstack([dpitch])

    return J

def jac_rpy_qut_numerical_mine(q):
    J = np.zeros((3, 4))
    eps = 1e-2
    q_ = copy.copy(q)
    rpy0_debug = np.array(tfs.euler_from_quaternion(q))
    rpy0 = np.array(myrpy(convert_wxyz2xyzw(q)))
    print(rpy0)
    print(rpy0_debug)
    for i in range(4):
        q_[i] += eps
        rpy1 = np.array(myrpy(convert_wxyz2xyzw(q_)))
        rpy1_debug = np.array(tfs.euler_from_quaternion(q_))

        print(rpy1)
        print(rpy1_debug)

        hoge = (rpy1 - rpy0)/eps
        J[:, i] = hoge
    return J


def jac_rpy_qut_numerical_tfs(q):
    J = np.zeros((3, 4))
    eps = 1e-6
    q_ = copy.copy(q)
    rpy0 = np.array(tfs.euler_from_quaternion(q))
    for i in range(4):
        q_[i] += eps
        rpy1 = np.array(tfs.euler_from_quaternion(q_))
        hoge = (rpy1 - rpy0)/eps
        J[:, i] = hoge
    return J

def convert_wxyz2xyzw(q):
    q_ = copy.copy(q)
    q_[0] = q[1]
    q_[1] = q[2]
    q_[2] = q[3]
    q_[3] = q[0]
    return q_

if __name__=='__main__':
    quat = tfs.quaternion_from_euler(0.19, 0.28, 0.38)
    quat_xyzw = convert_wxyz2xyzw(quat)
    print(myrpy(quat_xyzw))


    #print(jac_rpy_qut_numerical_tfs(quat))
    #jac_rpy_qut_numerical_mine(quat)

