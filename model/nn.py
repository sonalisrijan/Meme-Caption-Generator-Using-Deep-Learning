import torch
import torch.nn as nn

class HilarityNN(nn.Module):
    def __init__(self, num_inputs, num_classes, num_neurons_per_layer):
        super(HilarityNN, self).__init__()
        self.fc1 = nn.Linear(num_inputs, num_neurons_per_layer)
        self.act1 = nn.Sigmoid()
        self.fc2 = nn.Linear(num_neurons_per_layer, num_classes)
        self.act2 = nn.Sigmoid()
        
    def forward(self, input_):
        a1 = self.fc1(input_)
        h1 = self.act1(a1)
        a2 = self.fc2(h1)
        y = self.act2(a2)
        return y

class RelevanceNN(nn.Module):
    def __init__(self, num_inputs, num_classes, num_neurons_per_layer):
        super(RelevanceNN, self).__init__()
        self.fc1 = nn.Linear(num_inputs, num_neurons_per_layer)
        self.act1 = nn.Sigmoid()
        self.fc2 = nn.Linear(num_neurons_per_layer, num_classes)
        self.act2 = nn.Sigmoid()
        
    def forward(self, input_):
        a1 = self.fc1(input_)
        h1 = self.act1(a1)
        a2 = self.fc2(h1)
        y = self.act2(a2)
        return y