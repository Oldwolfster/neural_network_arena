import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Custom activation functions
class ShiftedReLU(nn.Module):
    def forward(self, x):
        return torch.maximum(torch.tensor(-1.0), x)

class ScaledSigmoid(nn.Module):
    def forward(self, x):
        return 2 * torch.sigmoid(x) - 1

# Networks with proper loss handling
class StandardNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2, 4)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(4, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return self.sigmoid(x)

class ModifiedNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2, 4)
        self.shifted_relu = ShiftedReLU()
        self.layer2 = nn.Linear(4, 1)
        # Note: We still need regular sigmoid for BCE loss
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.shifted_relu(x)
        x = self.layer2(x)
        return self.sigmoid(x)

    # Separate method for getting [-1,1] output
    def predict_scaled(self, x):
        return self.scaled_sigmoid(self.layer2(self.shifted_relu(self.layer1(x))))

def generate_data():
    np.random.seed(42)
    n_points = 1000

    X1 = np.random.randn(n_points//2, 2) * 0.5 + np.array([1, 1])
    X2 = np.random.randn(n_points//2, 2) * 0.5 + np.array([-1, -1])
    X = np.vstack([X1, X2])

    y = np.zeros(n_points)
    y[:n_points//2] = 1

    # Normalize inputs to [-1, 1] range
    X = (X - X.min()) / (X.max() - X.min()) * 2 - 1

    X = torch.FloatTensor(X)
    y = torch.FloatTensor(y).reshape(-1, 1)

    return X, y

def train_model(model, X, y, epochs=100):
    torch.manual_seed(42)  # Add seed for reproducibility
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCELoss()

    losses = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

    return losses

# Run experiment
rndSeed = 42
print("Generating data...")
X, y = generate_data()

print("\nTraining standard network...")
standard_net = StandardNet()
standard_losses = train_model(standard_net, X, y)

print("\nTraining modified network...")
modified_net = ModifiedNet()
modified_losses = train_model(modified_net, X, y)

# Plotting results
plt.figure(figsize=(15, 5))

# Plot 1: Training Losses
plt.subplot(1, 3, 1)
plt.plot(standard_losses, label='Standard')
plt.plot(modified_losses, label='Modified')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Plot 2: Decision Boundaries - Standard Net
plt.subplot(1, 3, 2)
x1_range = np.linspace(-1, 1, 100)
x2_range = np.linspace(-1, 1, 100)
X1, X2 = np.meshgrid(x1_range, x2_range)
grid = torch.FloatTensor(np.c_[X1.ravel(), X2.ravel()])
Z_standard = standard_net(grid).detach().numpy().reshape(X1.shape)

plt.contourf(X1, X2, Z_standard, levels=20, cmap='RdBu')
plt.scatter(X[:, 0], X[:, 1], c=y.reshape(-1), cmap='RdYlBu')
plt.title('Standard Network')
plt.grid(True)

# Plot 3: Decision Boundaries - Modified Net
plt.subplot(1, 3, 3)
Z_modified = modified_net(grid).detach().numpy().reshape(X1.shape)
plt.contourf(X1, X2, Z_modified, levels=20, cmap='RdBu')
plt.scatter(X[:, 0], X[:, 1], c=y.reshape(-1), cmap='RdYlBu')
plt.title('Modified Network')
plt.grid(True)

plt.tight_layout()
plt.show()

# Activation function comparison
x = torch.linspace(-2, 2, 200)

plt.figure(figsize=(15, 5))

# Plot standard ReLU
plt.subplot(1, 3, 1)
plt.plot(x.numpy(), nn.ReLU()(x).numpy(), label='Standard ReLU')
plt.title('Standard ReLU')
plt.grid(True)

# Plot shifted ReLU
plt.subplot(1, 3, 2)
plt.plot(x.numpy(), ShiftedReLU()(x).numpy(), label='Shifted ReLU')
plt.title('Shifted ReLU')
plt.grid(True)

# Plot both sigmoids
plt.subplot(1, 3, 3)
plt.plot(x.numpy(), nn.Sigmoid()(x).numpy(), label='Standard Sigmoid')
plt.plot(x.numpy(), ScaledSigmoid()(x).numpy(), label='Scaled Sigmoid')
plt.title('Sigmoid Comparison')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Print final accuracies with seed information
print("\nFinal Results:")
with torch.no_grad():
    standard_acc = ((standard_net(X) > 0.5) == y).float().mean()
    modified_acc = ((modified_net(X) > 0.5) == y).float().mean()

print(f"Standard Network Accuracy: {standard_acc:.4f}")
print(f"Modified Network Accuracy: {modified_acc:.4f}")