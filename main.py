import numpy as np

# Funções de ativação e derivadas
def relu(x):
    res = np.maximum(0, x)
    return res

def relu_derivative(x):
    res = (x > 0).astype(float)
    print(res)
    return res

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Função de perda cross-entropy
def cross_entropy_loss(y_true, y_pred):
    m = y_true.shape[0]
    return -np.sum(y_true * np.log(y_pred + 1e-9)) / m

def accuracy(y_true, y_pred):
    return np.mean(np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1))

# Inicialização da rede neural
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

    def forward(self, X):
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = relu(self.Z1)
        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = softmax(self.Z2)
        return self.A2

    def backward(self, X, y_true):
        m = X.shape[0]
        dZ2 = self.A2 - y_true
        dW2 = (self.A1.T @ dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * relu_derivative(self.Z1)
        dW1 = (X.T @ dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1

    def train(self, X, y, epochs=1000):
        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = cross_entropy_loss(y, y_pred)
            acc = accuracy(y, y_pred)
            self.backward(X, y)

            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1}, Loss: {loss:.4f}, Accuracy: {acc:.4f}")

# Exemplo de uso:
if __name__ == "__main__":
    # Gerando um dataset simples (exemplo: XOR expandido)
    from sklearn.datasets import make_moons
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OneHotEncoder

    X, y = make_moons(n_samples=1000, noise=0.1)
    y = y.reshape(-1, 1)

    encoder = OneHotEncoder(sparse_output=False)
    y_onehot = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

    nn = NeuralNetwork(input_size=2, hidden_size=16, output_size=2, learning_rate=0.1)
    nn.train(X_train, y_train, epochs=1000)

    y_pred_test = nn.forward(X_test)
    print("Test Accuracy:", accuracy(y_test, y_pred_test))
