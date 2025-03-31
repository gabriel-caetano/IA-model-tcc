import numpy as np

class Neuron:
    def __init__(self, input_size, activation_function=None, activation_derivative=None):
        """
        Generic neuron class for deep learning.

        :param input_size: Number of inputs the neuron accepts
        :param activation_function: Function to apply non-linearity (default: ReLU)
        :param activation_derivative: Derivative of the activation function for training
        """
        self.weights = np.random.randn(input_size)  # Random weight initialization
        self.bias = np.random.randn()  # Random bias
        self.activation_function = activation_function if activation_function else self.relu
        self.activation_derivative = activation_derivative if activation_derivative else self.relu_derivative
        self.last_input = None  # Store input for backpropagation
        self.last_output = None  # Store output for backpropagation

    def relu(self, x):
        """ReLU activation function."""
        return np.maximum(0, x)

    def relu_derivative(self, x):
        """Derivative of ReLU (needed for backpropagation)."""
        return np.where(x > 0, 1, 0)

    def forward(self, inputs):
        """
        Computes the neuron's output.

        :param inputs: Array of input values
        :return: Activated output
        """
        self.last_input = np.array(inputs)  # Store input for learning
        raw_output = np.dot(self.weights, self.last_input) + self.bias  # Weighted sum
        self.last_output = self.activation_function(raw_output)  # Apply activation
        return self.last_output

    def backward(self, dL_dout, learning_rate=0.01):
        """
        Performs backpropagation to update weights.

        :param dL_dout: Gradient of the loss with respect to the neuron’s output
        :param learning_rate: Learning rate for weight updates
        """
        d_out_d_raw = self.activation_derivative(self.last_output)  # Activation derivative
        d_raw_d_weights = self.last_input  # Inputs to neuron
        d_raw_d_bias = 1  # Bias derivative

        # Compute gradients
        dL_d_raw = dL_dout * d_out_d_raw
        dL_d_weights = dL_d_raw * d_raw_d_weights
        dL_d_bias = dL_d_raw * d_raw_d_bias

        # Update weights and bias
        # print("previous weights: ", self.weights)
        # print("new weights: ", learning_rate)
        # print("new weights: ", dL_d_weights)
        # print("new weights: ", learning_rate * dL_d_weights)
        self.weights -= learning_rate * dL_d_weights
        self.bias -= learning_rate * dL_d_bias


