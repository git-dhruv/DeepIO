import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch.optim as optim
from torchdiffeq import odeint_adjoint as odeint

### CREATING DATASET ###
def scale(x):
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    return scaler_X.fit_transform(x.reshape(-1,1))

def CreateTrainingData(dataFrame, Size=1000, StartPt=13000):
    # Extracting data from dataframe
    x,y,z = dataFrame['pose.position.x'].to_numpy(),dataFrame['pose.position.y'].to_numpy(),dataFrame['pose.position.z'].to_numpy()
    xA,yA,zA = dataFrame['psi'].to_numpy(),dataFrame['theta'].to_numpy(),dataFrame['phi'].to_numpy()
    t = dataFrame['Time'].to_numpy()
    u1, u2, u3, u4 = dataFrame['rpm_0'].to_numpy(), dataFrame['rpm_1'].to_numpy(), dataFrame['rpm_2'].to_numpy(), dataFrame['rpm_3'].to_numpy()

    StartPt = 13000
    t = t - t[StartPt]

    xTrain = t[StartPt:StartPt + Size].reshape(-1,1)
    yTrain = np.stack((scale(x[StartPt:StartPt + Size]),scale(xA[StartPt:StartPt + Size]),scale(y[StartPt:StartPt + Size]),scale(yA[StartPt:StartPt + Size]),scale(z[StartPt:StartPt + Size]),scale(zA[StartPt:StartPt + Size]),scale(u1[StartPt:StartPt + Size]),scale(u2[StartPt:StartPt + Size]),scale(u3[StartPt:StartPt + Size]),scale(u4[StartPt:StartPt + Size])),axis=1).reshape((Size,10))
    return xTrain, yTrain

def CreateBatch(yTrain, xTrain, size, batch_size, batch_time):
        
    s = torch.from_numpy(np.random.choice(np.arange(size-batch_time, dtype=np.int64), batch_size, replace=False))
    batch_y0 = yTrain[s,:]
    batch_x = xTrain[:batch_time] 
    batch_y = torch.stack([torch.Tensor(yTrain[s + i,:6]) for i in range(batch_time)], dim=0)

    return batch_y0, batch_x, batch_y

def TrainModel(yTrain, xTrain, size, batch_size, batch_time, FuncControl, path = '/TrainedModels/', niters=2000):

    parameters = FuncControl.parameters()
    optimizer = optim.Adadelta(params= parameters, lr=1.0, rho=0.9, eps=1e-06, weight_decay=0)

    LossHistory = []
    test_freq = 10
    for itr in range(1,niters +1):

        optimizer.zero_grad()    
        batch_y0, batch_x, batch_y = CreateBatch(yTrain, xTrain.flatten(), size, batch_size, batch_time)
        pred_y = odeint(FuncControl, torch.Tensor(batch_y0), torch.Tensor(batch_x), method='dopri5')
        
        loss = torch.mean(torch.abs(pred_y[:,:,:6] - torch.Tensor(batch_y)))
        loss.backward()
        optimizer.step()

        if itr % test_freq == 0:
            '''with torch.no_grad():
                pred_y = odeint(Func, torch.Tensor(y0[1,:]), t)
                loss = torch.norm(pred_y - torch.Tensor(y[:,1,:]))'''
            print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
            LossHistory.append(loss.item())

    torch.save({
    'epoch': itr,
    'model_state_dict': FuncControl.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
    }, path)
    
    return FuncControl, LossHistory

if __name__ == '__main__':

    # Converting dataset in rosbag form to dataframe
    from dataloader import dataloader
    from NODE import ODEFunc

    tmp = dataloader("data/Oval/")
    imu, rotor, mocap = tmp.runPipeline()
    tmp.homogenizeData()
    dataFrame = tmp.ConcatData

    size = 1000
    batch_time = 5
    batch_size = 500
    niters = 2000
    
    xTrain, yTrain = CreateTrainingData(dataFrame, size)
    FuncControl = ODEFunc()
    FuncControl, LossHistory = TrainModel(yTrain, xTrain, size, batch_size, batch_time, FuncControl,path='TrainingDynamics/TrainedModels/Save.pt', niters=2)
    plt.plot(LossHistory)
    plt.show()