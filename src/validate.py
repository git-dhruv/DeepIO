import numpy as np
import matplotlib.pyplot as plt
from dataloader import *
from dynamicsSim import *
from numpy import sin, cos
import tqdm
from copy import deepcopy
from estimate_rot import ukfPipeline
from complementary import *
import torch
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from os.path import join as opj

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LSTM
# import os

# os.environ['KMP_DUPLICATE_LIB_OK']='True'

from imuModel import Model



# torch.set_default_tensor_type('torch.c/ddduda.FloatTensor')

dataDir = r"data/clover"
loadDataUtil = dataloader(dataDir)
loadDataUtil.runPipeline()
loadDataUtil.homogenizeData()
gyro, acc, rpm, mocap, q, t = loadDataUtil.convertDataToIndividualNumpy()        
Gtruth = mocap.copy()        

#____________Rotate the Motion Capture to IMU frame from Body Frame_______#
R_imutoBody= np.array([ [0, -1, 0],
                        [1,  0, 0],
                        [0,  0, 1]])

mocap = R_imutoBody @ mocap
Gtruth = R_imutoBody@ Gtruth
eulers = []
quats = q.T
for i in range(quats.shape[0]):
    R_temp = R_imutoBody@Rotation.from_quat(quats[i, :]).as_matrix()
    quats[i, :] = Rotation.from_matrix(R_temp).as_quat().flatten()
    quat = Quaternion(scalar = quats[i, -1], vec = quats[i, 0:3])
    eulers.append(quat.euler_angles())
eulers = np.array(eulers)

device = 'cuda:0'
# device = 'cpu'

##################################
## Define network and optimizer ##
##################################

print('[SETUP] Establishing model and optimizer.')
model = Model(device).to(device)

model.load_state_dict(torch.load(r"C:\Users\dhruv\Desktop\Penn\Sem2\ESE650\FinalProject\DeepIO\model_4000.pth"))
model.eval()

roll = []
pitch = []

for i in tqdm.tqdm(range(500,25000)):
    dt = t[i] - t[i-1]
    # acc[:,i] = acc[:,i]/np.linalg.norm(acc[:,i])
    
    measurementPacket = np.array([float(acc[0,i]),float(acc[1,i]),float(acc[2,i]),
                                        float(gyro[0,i]),float(gyro[1,i]),float(gyro[2,i])])
    measurementPacket = torch.tensor(measurementPacket).to(device)
    pred = model(measurementPacket.float())
    # print(pred)
    angs = pred.cpu().detach().numpy().flatten()
    roll.append(float(angs[0]))
    pitch.append(float(angs[1]))


# plt.plot(roll)
plt.plot(eulers[500:25000,1])
plt.plot(pitch)
# plt.plot(eulers)
plt.show()
