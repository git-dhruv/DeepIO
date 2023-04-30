
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.nn import LSTM

# class Model(nn.Module):
#     def __init__(self, device):
#         super().__init__()   
#         #Input -> g3 ac3 and previous angles -> 3 + dt    -> 10
#         self.fc1 = nn.Linear(7, 56)
#         self.fc2 = nn.Linear(56,128)
#         # Added LSTM
#         self.lstm = LSTM(input_size=128, hidden_size=48,
#                          num_layers=5, dropout=0.3)
#         self.fc3 = nn.Linear(48, 30, bias=False)
#         self.fc4 = nn.Linear(30,3)
#         # Added another FC Layer

#     def forward(self, x):
#         # measurementPacket,eulers[i-1,:],dt
#         x = self.fc1(x)
#         x = F.leaky_relu(x)
#         x = self.fc2(x)
#         x = F.leaky_relu(x)
#         x = x.unsqueeze(0)
#         # LSTM
#         x, _ = self.lstm(x)
#         x = x.view(-1, 1).flatten()

#         x = self.fc3(x)
#         x = F.relu(x)
#         x = self.fc4(x)
        
#         return x
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LSTM, RNN

class Model(nn.Module):
    def __init__(self, device):
        super().__init__()   
        #Input -> g3 ac3 and previous angles -> 3 + dt    -> 10
        # self.bn1 = nn.BatchNorm1d(7)

        self.fc1 = nn.Linear(6, 20)
        self.fc2 = nn.Linear(20,20)
        # Added LSTM
        self.lstm = LSTM(input_size=20, hidden_size=48,
                         num_layers=5, batch_first=True)
        self.rnn = RNN(input_size=48, hidden_size=24, num_layers=2, batch_first=True)
        self.dropout = nn.Dropout(p=0.2)
        self.fc4 = nn.Linear(24,2)
        # Added another FC Layer

    def forward(self, x):
        # measurementPacket,eulers[i-1,:],dt
        # x = self.bn1(x)
        x = x.unsqueeze(0).float()
        # print(x.size())
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        # LSTM
        x, _ = self.lstm(x)
        # RNN
        x, _ = self.rnn(x)
        x= F.relu(x)
        # x = x.view(-1, 1).flatten()
        x = self.fc4(x)
        
        return x
