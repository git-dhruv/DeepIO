# %%
import numpy as np
import matplotlib.pyplot as plt
from dataloader import *
from dynamicsSim import *
from numpy import sin, cos

import tqdm
import pypose

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# %%
dataDir = r'C:\Users\dhruv\Desktop\Penn\Sem2\ESE650\FinalProject\DeepIO\data\clover'
loadDataUtil = dataloader(dataDir)
loadDataUtil.runPipeline()
loadDataUtil.homogenizeData()
gyro, acc, rpm, mocap, q, t = loadDataUtil.convertDataToIndividualNumpy()

##--Rotate the Motion Capture to IMU frame from Body Frame --#
R_imutoBody= np.array([[0, -1, 0],
                    [1, 0, 0],
                    [0,0,1]])


# %%
import torch
mocap = R_imutoBody@mocap
pos =torch.tensor(mocap[:,0]).float()    
r = pypose.mat2SO3(Rotation.from_quat([q[:,0]]).as_matrix()).float()
v = torch.zeros(3).float()
dt = torch.tensor([0.002]).float() 
integrator = pypose.module.IMUPreintegrator(pos, r, v)
# acc = acc - acc[:,:50].mean(axis=1).reshape(-1,1)
# print(acc[:,0].flatten())
# %%
x,y,z = [],[],[]
for i in range(1,10000):
    dt = torch.tensor(t[i]-t[i-1]).unsqueeze(0)
    states = integrator(dt.float(), torch.tensor(gyro[:,i].flatten()).float(), torch.tensor(acc[:,i].flatten()).float())
    x.append(float(states['pos'][0].numpy()[0,0]))
    y.append(float(states['pos'][0].numpy()[0,1]))
    z.append(float(states['pos'][0].numpy()[0,2]))


# %%
print(x)
plt.plot(x);
plt.plot(mocap[0,:1000])
plt.plot(mocap[1,:1000])
plt.plot(mocap[2,:1000])


plt.show()


