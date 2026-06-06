from torch_geometric.nn import SAGEConv
from torch_geometric.nn import global_mean_pool
import torch.nn.functional as F
import torch
import torch.nn as nn

class FraudGNN(torch.nn.Module):
    def __init__(self, in_channels=12, hidden_channels=64, out_channels=2):
        super(FraudGNN, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index,batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        h = self.conv3(x, edge_index)
        h = F.relu(h)
        x = global_mean_pool(h, batch) 
        out = self.classifier(x)
        return out, h
    
