#### NODE #####

import torch.nn as nn

class ODEFunc(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 35),
            nn.LeakyReLU(),
            nn.Linear(35, 35),
            nn.LeakyReLU(),
            nn.Linear(35, 10),
        )
        self.net = self.net.float()
        self.net.apply(self._apply_wt_init)

    def forward(self, t, y):
        return self.net(y)
    
    def _apply_wt_init(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.normal_(layer.weight, mean=0, std=0.1)
            nn.init.constant_(layer.bias, val=0)