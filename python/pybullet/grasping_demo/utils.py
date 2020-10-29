import pybullet as p

CLIENT = 0
def get_pose(body):
    return p.getBasePositionAndOrientation(body, physicsClientId=CLIENT)

def get_quat(body):
    return get_pose(body)[1] # [x,y,z,w]

def get_point(body):
    return get_pose(body)[0]

def set_pose(body, pose):
    (point, quat) = pose
    p.resetBasePositionAndOrientation(body, point, quat, physicsClientId=CLIENT)

def set_point(body, point):
    set_pose(body, (point, get_quat(body)))

def quat_from_euler(euler):
    return p.getQuaternionFromEuler(euler)

def z_rotation(theta):
    return quat_from_euler([0, 0, theta])

def set_quat(body, quat):
    set_pose(body, (get_point(body), quat))

def set_4dpose(body, values): #x, y, z, theta
    x, y, z, theta = values
    set_point(body, (x, y, z))
    set_quat(body, z_rotation(theta))

def set_6dpose(body, pos, rpy):
    set_point(body, pos)
    set_quat(body, quat_from_euler(rpy))

def set_zrot(body, theta):
    set_quat(body, z_rotation(theta))

