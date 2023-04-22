# %%
from dataloader import *

# %%
import matplotlib.pyplot as plt
import numpy as np
tmp = dataloader("../data/clover")
imu, rotor, mocap = tmp.runPipeline()
    

# %%
x,y,z = mocap['pose.position.x'].to_numpy(),mocap['pose.position.y'].to_numpy(),mocap['pose.position.z'].to_numpy()


# %%
print(mocap['Time'][0]-imu['Time'][0])
print()
# imu['linear_acceleration.x']

# %%
from ahrs.filters import Madgwick, EKF
gyrox,gyroy,gyroz = imu['angular_velocity.x'].to_numpy(),imu['angular_velocity.y'].to_numpy(),imu['angular_velocity.z'].to_numpy()
accx,accy,accz = imu['linear_acceleration.x'].to_numpy(),imu['linear_acceleration.y'].to_numpy(),imu['linear_acceleration.z'].to_numpy()
# madgwick = Madgwick(gyr=, acc=acc_data)     

# %%
gyro = np.column_stack((np.column_stack((gyrox.reshape(-1,1),gyroy.reshape(-1,1))),gyroz.reshape(-1,1)))
acc = np.column_stack((np.column_stack((accx.reshape(-1,1),accy.reshape(-1,1))),accz.reshape(-1,1)))
mag = np.zeros_like(acc)
# mag[:,0] += 254.3
mag[:,1] += 1
# mag[:,2] += 558.6


# %%


# %%
from scipy.spatial.transform import Rotation as R
R_w_ned = np.array([
    [1., 0., 0.],
    [0., -1., 0.],
    [0., 0., -1.]])
t_w_ned = np.array([0., 0., 0.])

# rotate from body to imu frame
R_b_i = np.array([
    [0., -1., 0.],
    [1., 0., 0.],
    [0., 0., 1.]])

# %%


# %%


# %%
from dynamicsSim import *
dyn = dynamics()

i = 29000
dt = rotor['Time'][i]-rotor['Time'][i-1]
state = [x[i],y[i],z[i],(x[i]-x[i-1])/dt,(y[i]-y[i-1])/dt,(z[i]-z[i-1])/dt,mocap['pose.orientation.x'][i],mocap['pose.orientation.y'][i],mocap['pose.orientation.z'][i],mocap['pose.orientation.w'][i],0,0,0]
state2 = dyn.propogateDynamics(np.array(state),np.array([rotor['rpm_0'][i],rotor['rpm_1'][i],rotor['rpm_2'][i],rotor['rpm_3'][i]]),dt)
i += 1
state = [x[i],y[i],z[i],(x[i]-x[i-1])/dt,(y[i]-y[i-1])/dt,(z[i]-z[i-1])/dt,mocap['pose.orientation.x'][i],mocap['pose.orientation.y'][i],mocap['pose.orientation.z'][i],mocap['pose.orientation.w'][i],0,0,0]
print(np.round(state,3))
print(np.round(state2,3))
print(dt)



# %%
