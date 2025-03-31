import numpy as np
from dnn.neuron import Neuron
import matplotlib.pyplot as plt

### Test Neuron with a single execution

# Example usage
neuron = Neuron(input_size=3)
inputs = [0.5, -1.2, 2.3]

print("Neuron Inputs:", inputs)
print("Original Weights:", neuron.weights)
print("Original Bias:", neuron.bias)

output = neuron.forward(inputs)

# Simulating a backward pass with a loss gradient
loss_gradient = 1.0  # Example loss gradient from the next layer
neuron.backward(loss_gradient)
print("Neuron Output:", output)
print("Updated Weights:", neuron.weights)
print("Updated Bias:", neuron.bias)

### Test neuron with a test dataset

# Generate dataset (random points)
np.random.seed(42)  # For reproducibility
num_samples = 100

# Inputs: two random features between -1 and 1
X = np.random.uniform(-1, 1, (num_samples, 2))

# Labels: 1 if x1 + x2 > 0, else 0
y = (X[:, 0] + X[:, 1] > 0).astype(int)

# Plot the dataset
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="bwr", edgecolors="k")
plt.xlabel("Feature 1 (x1)")
plt.ylabel("Feature 2 (x2)")
plt.title("Synthetic Binary Classification Dataset")
plt.colorbar(label="Class (0 or 1)")
plt.show()
