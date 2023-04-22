import matplotlib.pyplot as plt
from math import *
import numpy as np
from estimate_rot import ukfPipeline
from dataloader import *




tmp = dataloader("../data/clover")
imu, rotor, mocap = tmp.runPipeline()
gyrox,gyroy,gyroz = imu['angular_velocity.x'].to_numpy(),imu['angular_velocity.y'].to_numpy(),imu['angular_velocity.z'].to_numpy()
accx,accy,accz = imu['linear_acceleration.x'].to_numpy(),imu['linear_acceleration.y'].to_numpy(),imu['linear_acceleration.z'].to_numpy()
gyro = np.column_stack((np.column_stack((gyrox.reshape(-1,1),gyroy.reshape(-1,1))),gyroz.reshape(-1,1)))
acc = np.column_stack((np.column_stack((accx.reshape(-1,1),accy.reshape(-1,1))),accz.reshape(-1,1)))


R_imu_to_ned = np.array([[-1,0,0],
                                [0,-1,0],
                                [0,0,1]])

T = imu['Time'].to_numpy()
gyro = R_imu_to_ned@gyro.T
acc = R_imu_to_ned@acc.T

print(acc[:,0])
sol = ukfPipeline(acc[:,:1000], gyro[:,:1000], T[:1000])
stateVector = sol.runPipeline()
# roll, pitch, yaw are numpy arrays of length T
r,p,y = sol.quat2rpy(stateVector[:4, :])

plt.plot(r)
plt.plot(p)
plt.plot(y)
plt.show()
