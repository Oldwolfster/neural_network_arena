import numpy as np
import torch
import torch.nn as nn

# Paper's proposed modifications
def scaled_sigmoid(x):
    return 2 * torch.sigmoid(x) - 1

def shifted_relu(x):
    return torch.maximum(torch.tensor(-1.0), x)

# Simple test
x = torch.linspace(-2, 2, 100)
standard_relu = nn.ReLU()(x)
modified_relu = shifted_relu(x)

# You can plot these to see the difference:
import matplotlib.pyplot as plt
plt.plot(x.numpy(), standard_relu.numpy(), label='Standard ReLU')
plt.plot(x.numpy(), modified_relu.numpy(), label='Shifted ReLU')
plt.legend()
plt.show()