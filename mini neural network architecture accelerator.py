import torch
import torch.nn as nn

# Neural Network Model
class NeuralNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        # Hidden Layer: 3 inputs -> 3 neurons
        self.hidden = nn.Linear(3, 3)

        # ReLU Activation
        self.relu = nn.ReLU()

        # Output Layer: 3 neurons -> 1 output
        self.output = nn.Linear(3, 1)

    def forward(self, x):

        hidden_out = self.hidden(x)

        relu_out = self.relu(hidden_out)

        final_out = self.output(relu_out)

        return hidden_out, relu_out, final_out


# Create model
model = NeuralNetwork()

# Sample inputs
X = torch.tensor([
    [1.0, 2.0, 3.0],
    [2.0, 3.0, 4.0],
    [3.0, 4.0, 5.0]
])

# Run model
hidden_out, relu_out, final_out = model(X)

# Display results
print("================================")
print("MINI NEURAL NETWORK ACCELERATOR")
print("================================")

print("\nInput:")
print(X)

print("\nHidden Layer Output:")
print(hidden_out.detach())

print("\nReLU Layer Output:")
print(relu_out.detach())

print("\nFinal Output Layer:")
print(final_out.detach())

print("\nHidden Layer Weights:")
print(model.hidden.weight.detach())

print("\nHidden Layer Bias:")
print(model.hidden.bias.detach())

print("\nOutput Layer Weights:")
print(model.output.weight.detach())

print("\nOutput Layer Bias:")
print(model.output.bias.detach()

