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
from sklearn.preprocessing import MinMaxScaler

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

basedir = r'src'
expname = datetime.now().strftime('d%m_%d_t%H_%M')
workspace = opj(basedir, "logs/", expname)
wkspc_ctr = 2

while os.path.exists(workspace):
    workspace = opj(basedir,"logs/", expname+f'_{str(wkspc_ctr)}')
    wkspc_ctr += 1
os.makedirs(workspace)
writer = SummaryWriter(workspace)

measurements = np.vstack((acc,gyro)).T
scaler_X = MinMaxScaler(feature_range=(0, 1))
measurements = scaler_X.fit_transform(measurements)
eulers = scaler_X.fit_transform(eulers)



##################################
## Define network and optimizer ##
##################################

print('[SETUP] Establishing model and optimizer.')
model = Model(device).to(device)
# optimizer = torch.optim.SGD(model.parameters(), lr=5e-6, momentum=0.9)
optimizer = torch.optim.Adam(model.parameters(), lr=1)
# optimizer = torch.optim.Adadelta(params=model.parameters(), lr=1e-3)

# model.load_state_dict(torch.load("C:\Dhruv Personal Files\RL\\agile_flight_dev_env\\agile_flight\model_120.pth"))
# model.load_state_dict(torch.load(r"C:\Users\dhruv\Desktop\Penn\Sem2\ESE650\FinalProject\DeepIO\model.pth"))

N_eps = 10000
itrs = 0
print(f'[TRAIN] Training for {N_eps} epochs.')
total_itr = 0



    #### NODE #####
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint_adjoint as odeint
import numpy as np
import matplotlib.pyplot as plt

method = 'dopri5'
batch_time = 15
batch_size = TrainSize-100
niters = 2000
test_freq = 10
viz = 'store_true'
adjoint = 'store_true'

def mini_batch(yTrain, xTrain, size, batch_size, batch_time):
        
    s = torch.from_numpy(np.random.choice(np.arange(size-batch_time, dtype=np.int64), batch_size, replace=False))
    batch_y0 = xTrain[s,:4]
    batch_x = xTrain[:batch_time,4] 
    batch_y = torch.stack([torch.Tensor(xTrain[s + i,:4]) for i in range(batch_time)], dim=0)
    batch_yy = yTrain[s,:]
    batch_yy1 = yTrain[s+1,:]

    return batch_y0, batch_x, batch_y, batch_yy, batch_yy1

def TrainNODENetwork(yTrain, xTrain, size, batch_size, batch_time, FuncControl, FuncDynamics, niters=10000,):

    parameters = list(FuncControl.parameters()) + list(FuncDynamics.parameters())
    #optimizerD = optim.Adam(params=parameters, lr=1e-3)
    optimizer = optim.Adadelta(params=parameters, lr=1.0, rho=0.9, eps=1e-06, weight_decay=0) #

    for itr in range(1,niters +1):
        optimizer.zero_grad()    
        batch_y0, batch_x, batch_y, batch_yy, batch_yy1 = mini_batch(yTrain, xTrain, size, batch_size, batch_time)
        pred_y = odeint(FuncControl, torch.Tensor(batch_y0), torch.Tensor(batch_x))
        
        XDynamics = torch.cat((torch.Tensor(batch_y0), torch.Tensor(batch_yy)), 1) #pred_y[1,:,:]
        pred_yy = FuncDynamics(XDynamics)
        lossD = torch.mean(torch.abs(pred_yy - torch.Tensor(batch_yy1)))
        #lossC = torch.mean(torch.abs(pred_y - torch.Tensor(batch_y)))

        loss = lossD# + lossC
        loss.backward()
        optimizer.step()

        if itr % test_freq == 0:
            '''with torch.no_grad():
                pred_y = odeint(Func, torch.Tensor(y0[1,:]), t)
                loss = torch.norm(pred_y - torch.Tensor(y[:,1,:]))'''
            print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
    
    return FuncControl, FuncDynamics

class ODEFunc(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 30),
            nn.ReLU(),
            nn.Linear(30, 30),
            nn.ReLU(),
            nn.Linear(30, 4),
        )
        self.net = self.net.float()
        self.net.apply(self._apply_wt_init)

    def forward(self, t, y):
        return self.net(y)
    
    def _apply_wt_init(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.normal_(layer.weight, mean=0, std=0.1)
            nn.init.constant_(layer.bias, val=0)

class Func2(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 35),
            nn.Tanh(),
            nn.Linear(35, 35),
            nn.Tanh(),
            nn.Linear(35, 6),
        )
        self.net = self.net.float()
        self.net.apply(self._apply_wt_init)

    def forward(self, y):
        return self.net(y)
    
    def _apply_wt_init(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.normal_(layer.weight, mean=0, std=0.1)
            nn.init.constant_(layer.bias, val=0)

FuncControl = ODEFunc()
FuncDynamics = Func2()

FuncControl, FuncDynamics = TrainNODENetwork(train_y, train_x, TrainSize, batch_size, batch_time, FuncControl, FuncDynamics)








"""
















size = 100000

TrainSize = 80000
batch_time = 15
batch_size = TrainSize-100

eulers = torch.tensor(eulers).to(device)

measurements = torch.tensor(measurements).to(device)
for ep in range(N_eps):
    ep_loss = 0
    if ep%100==0:
        path = ""
        torch.save(model.state_dict(), path+f"model_{ep}.pth")
    
    # for i in tqdm.tqdm(range(500,5000)):
        # if i%20!=0:
        #     continue
    s = torch.from_numpy(np.random.choice(np.arange(size-batch_time, dtype=np.int64), batch_size, replace=False))
    
    batch_x = (measurements[s,:]).float()
    batch_y = eulers[s,:2].unsqueeze(0).float()

    # dt = t[i] - t[i-1]
    # acc[:,i] = acc[:,i]/np.linalg.norm(acc[:,i])
    # measurementPacket = np.array([float(acc[0,i]),float(acc[1,i]),float(acc[2,i]),
    #                                     float(gyro[0,i]),float(gyro[1,i]),float(gyro[2,i]),
    #                                     float(dt)])
    # measurementPacket = torch.tensor(measurementPacket).to(device)
    optimizer.zero_grad()
    
    pred = model(batch_x)
    loss = F.mse_loss(pred, batch_y)
    ep_loss = loss

    loss.backward()
    optimizer.step()

    print(f'[TRAIN] Completed ep {ep}/{N_eps-1}, ep_loss = {ep_loss:.6f}')

    writer.add_scalar('train/loss', ep_loss, ep)
    writer.flush()

print(f'[TRAIN] Training complete.')
path = ""
torch.save(model.state_dict(), path+"model.pth")

