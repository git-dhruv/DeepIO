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


class Model(nn.Module):
    def __init__(self, device):
        super().__init__()   
        #Input -> g3 ac3 and previous angles -> 3 + dt    -> 10
        self.fc1 = nn.Linear(10, 128, bias=False)
        # Added LSTM
        self.lstm = LSTM(input_size=128, hidden_size=48,
                         num_layers=5, dropout=0.3)
        self.fc3 = nn.Linear(48, 30, bias=False)
        self.fc4 = nn.Linear(30,3)
        # Added another FC Layer

    def forward(self, x):
        # measurementPacket,eulers[i-1,:],dt
        x = self.fc1(x)
        x = F.leaky_relu(x)

        x = x.unsqueeze(0)
        # LSTM
        x, _ = self.lstm(x)
        x = x.view(-1, 1).flatten()

        x = self.fc3(x)
        x = F.sigmoid(x)
        x = self.fc4(x)
        
        return x




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

basedir = r'src'
expname = datetime.now().strftime('d%m_%d_t%H_%M')
workspace = opj(basedir, "logs/", expname)
wkspc_ctr = 2

while os.path.exists(workspace):
    workspace = opj(basedir,"logs/", expname+f'_{str(wkspc_ctr)}')
    wkspc_ctr += 1
os.makedirs(workspace)
writer = SummaryWriter(workspace)

##################################
## Define network and optimizer ##
##################################

print('[SETUP] Establishing model and optimizer.')
model = Model(device).to(device)
# model = ConvNet(device=device)

# optimizer = torch.optim.SGD(model.parameters(), lr=5e-6, momentum=0.9)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)

# model.load_state_dict(torch.load("C:\Dhruv Personal Files\RL\\agile_flight_dev_env\\agile_flight\model_120.pth"))

N_eps = 4
itrs = 0
print(f'[TRAIN] Training for {N_eps} epochs.')
total_itr = 0
# eulers = torch.tensor(eulers).to(device)
for ep in range(N_eps):
    ep_loss = 0
    path = ""
    torch.save(model.state_dict(), path+f"model_{ep}.pth")
    for i in tqdm.tqdm(range(1,t.shape[0]-1000)):
        dt = t[i] - t[i-1]
        measurementPacket = np.array([float(acc[0,i]),float(acc[1,i]),float(acc[2,i]),
                                            float(gyro[0,i]),float(gyro[1,i]),float(gyro[2,i]),
                                            float(eulers[i-1,0]),
                                            float(eulers[i-1,1]),
                                            float(eulers[i-1,2]),
                                            float(dt)])
        measurementPacket = torch.tensor(measurementPacket).to(device)
        
        optimizer.zero_grad()
        pred = model(measurementPacket.float())
        loss = F.mse_loss(pred, torch.tensor(eulers[i,:]).to(device).float())
        ep_loss += loss

        loss.backward()
        optimizer.step()

        if i % 20000 == 0:
            tqdm_text = f'Loss = {loss:4f}'
            print(tqdm_text)
            # tqdm.write(tqdm_text)
        itrs = itrs+1

    if total_itr == 0:
        total_itr = itrs

    ep_loss /= total_itr

    print(f'[TRAIN] Completed ep {ep}/{N_eps-1}, ep_loss = {ep_loss:.6f}')

    writer.add_scalar('train/loss', ep_loss, ep)
    writer.flush()

print(f'[TRAIN] Training complete.')
path = ""
torch.save(model.state_dict(), path+"model.pth")

