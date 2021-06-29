import torch
import torch.nn as nn
import torch.nn.functional as F


torch.set_num_threads(1)


class Net(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=120):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        #self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        #self.bn2 = torch.nn.BatchNorm1d(hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, hidden_dim)
        #self.bn3 = torch.nn.BatchNorm1d(hidden_dim)
        self.fc4 = torch.nn.Linear(hidden_dim, hidden_dim)
        #self.bn4 = torch.nn.BatchNorm1d(hidden_dim)
        self.fc5 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float)
        x = self.fc1(x)
        #x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        #x = self.bn2(x)
        x = F.relu(x)
        x = self.fc3(x)
        #x = self.bn3(x)
        x = F.relu(x)
        x = self.fc4(x)
        #x = self.bn4(x)
        x = F.relu(x)
        x = self.fc5(x)
        return x
