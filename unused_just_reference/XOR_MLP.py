import torch
import torch.nn as nn
import torch

# XOR inputs
inputs = torch.tensor([[0.0, 0.0],
                       [0.0, 1.0],
                       [1.0, 0.0],
                       [1.0, 1.0]])

# XOR outputs
targets = torch.tensor([[0.0],
                        [1.0],
                        [1.0],
                        [0.0]])

from torch.utils.data import TensorDataset, DataLoader

# Create dataset
dataset = TensorDataset(inputs, targets)

# Create data loader
data_loader = DataLoader(dataset, batch_size=4, shuffle=True)



import torch.nn as nn
import torch.optim as optim

class GPT_XOR(nn.Module):
    def __init__(self):
        super(GPT_XOR, self).__init__()
        self.hidden = nn.Linear(2, 2)  # Input: 2, Hidden: 2
        self.output = nn.Linear(2, 1)  # Hidden: 2, Output: 1
        self.activation = nn.Tanh()  # Non-linear activation

        for layer in self.children():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)


    def forward(self, x):
        x = self.activation(self.hidden(x))  # Hidden layer with activation
        x = self.output(x)  # Output layer
        return x

    def train_model(self, data_loader, epochs=100,              learning_rate=0.1):
        # Loss function and optimizer
        criterion = nn.MSELoss()                    # Mean Squared Error for binary regression
        optimizer = optim.SGD(self.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            total_loss = 0
            for inputs, targets in data_loader:
                optimizer.zero_grad()  # Zero gradients
                outputs = self(inputs)  # Forward pass
                loss = criterion(outputs, targets)  # Compute loss
                loss.backward()  # Backward pass
                optimizer.step()  # Update weights
                total_loss += loss.item()
                print(f"Epoch {epoch+1}, Loss: {loss.item()}")

            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")
def evaluate_model(model, data_loader):
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():  # Disable gradient calculations
        for inputs, targets in data_loader:
            outputs = model(inputs)  # Forward pass
            predictions = (outputs > 0).float()  # Convert outputs to binary decisions
            print(f"Inputs: {inputs.numpy()}, Predictions: {predictions.numpy()}, Targets: {targets.numpy()}")


model = GPT_XOR()
model.train_model(data_loader, epochs=100, learning_rate=0.5)

evaluate_model(model, data_loader)

