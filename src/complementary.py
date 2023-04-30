# %% Imports

import numpy as np
from numpy.linalg import norm
from scipy.spatial.transform import Rotation


# %%
def calcBlend(error):
    alpha = 1
    if(error>0.1):
        alpha = -10*error+ 2
    if error>0.2:
        alpha = 0.0
    return alpha

def rotation_update(R_prev, R_current):
    R = R_prev*R_current
    quat = R.as_quat()
    quat = quat/np.linalg.norm(quat)
    return Rotation.from_quat(quat)
   
def complementary_filter_update(initial_rotation, angular_velocity, linear_acceleration, dt):
    """
    Implements a complementary filter update

    :param initial_rotation: rotation_estimate at start of update
    :param angular_velocity: angular velocity vector at start of interval in radians per second
    :param linear_acceleration: linear acceleration vector at end of interval in meters per second squared
    :param dt: duration of interval in seconds
    :return: final_rotation - rotation estimate after update
    """
    #----Incorporating GyroScope updates----#
    gyro = Rotation.from_rotvec(angular_velocity*dt)
    R_updated = rotation_update(initial_rotation,gyro)

    #----Accelerometer----#
    linear_acceleration = R_updated.as_matrix()@ linear_acceleration / 9.81
    normalized_acc = linear_acceleration/np.linalg.norm(linear_acceleration)
    ax,ay,az = normalized_acc.flatten()
    q0 = np.sqrt((1+az)/2)
    q3 = 0
    q2 = -ax/np.sqrt(2*(1+az))
    q1 = ay/np.sqrt(2*(1+az))
    q_acc = np.array([q1,q2,q3,q0]).reshape(-1,1)
        
    #----Update----#
    q_null = np.array([0,0,0,1]).reshape(-1,1)
    error = abs(np.linalg.norm(linear_acceleration)-1)
    alpha = calcBlend(error) 

    q_correction = (1-alpha)*q_null + alpha*(q_acc)
    q_correction = q_correction/np.linalg.norm(q_correction)
    return rotation_update(Rotation.from_quat(q_correction.flatten()),R_updated)