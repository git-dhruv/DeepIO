## We will first access training data from saved models and then use it to test the model 
import matplotlib.pyplot as plt
from NODE import ODEFunc
import torch.optim as optim
import torch
from dataloader import dataloader
from sklearn.preprocessing import MinMaxScaler
from torchdiffeq import odeint
from Training import CreateTrainingData

PATH = "TrainedModels/save.pt"

# Invoking saved model

model = ODEFunc()
parameters = model.parameters()
optimizer = optim.Adadelta(params= parameters, lr=1.0, rho=0.9, eps=1e-06, weight_decay=0)
checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

model.eval(True)

tmp = dataloader("data/Oval/")
imu, rotor, mocap = tmp.runPipeline()
tmp.homogenizeData()
dataFrame = tmp.ConcatData

size = 1000
batch_time = 5
batch_size = 500
niters = 2000

xTrain, yTrain = CreateTrainingData(dataFrame, size)
ax = plt.figure().add_subplot(projection='3d')
ax.plot(yTrain[:,0],yTrain[:,2],yTrain[:,4], color='red')

scaler_X = MinMaxScaler(feature_range=(0, 1))

for i in range(0,size-batch_time):
    y0 = yTrain[i,:]
    pred_u = odeint(model, torch.Tensor(y0.flatten()), torch.Tensor(scaler_X.fit_transform(xTrain[i:i+batch_time]).flatten()))
    pu = pred_u.detach().numpy()
    y0 =  scaler_X.fit_transform(pu[1,:].reshape(-1,1))
    ax.scatter(y0[0], y0[2],y0[4], s=2,c='b')

plt.show()

## ONLINE TRAINING
model.train(True)
