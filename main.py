import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_circles

X, y = make_circles(n_samples=100, noise=0.1, factor=0.3, random_state = 0)
X = X.T
y = y.reshape((1, y.shape[0]))
print("dimention de X:", X.shape)
print("dimention de y:", y.shape)


def init(n0, n1, n2):
    W1 = np.random.randn(n1, n0)
    W2 = np.random.randn(n2, n1)
    b1 = np.random.randn(n1, 1)
    b2 = np.random.randn(n2, 1)
    params = {
        "W1": W1,
        "W2": W2,
        "b1": b1,
        "b2": b2
    }
    return params


def forward_propagation(X, params):
    W1 = params["W1"]
    W2 = params["W2"]
    b1 = params["b1"]
    b2 = params["b2"]
    
    Z1 = W1.dot(X) + b1
    A1 = 1 / (1 + np.exp(-Z1))
    Z2 = W2.dot(A1) + b2
    A2 = 1 / (1 + np.exp(-Z2))

    activations = {
        "A1": A1,
        "A2": A2
    }
    return activations

def back_propagation(X, y, activations, params):
    A1 = activations["A1"]
    A2 = activations["A2"]
    W2 = params["W2"]
    m = y.shape[1]

    dZ2 = A2 - y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.dot(W2.T, dZ2) * A1 * (1 - A1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)

    gradients = {
        "dW2": dW2,
        "dW1": dW1,
        "db2": db2,
        "db1": db1
    }
    return gradients
 
def update(params, gradients, learning_rate):
    params["W1"] = params["W1"] - learning_rate*gradients["dW1"]
    params["W2"] = params["W2"] - learning_rate*gradients["dW2"]
    params["b1"] = params["b1"] - learning_rate*gradients["db1"]
    params["b2"] = params["b2"] - learning_rate*gradients["db2"]

    return params

def log_loss(A, y):
    L = 1/len(y) * np.sum(-y*np.log(A) - (1-y) * np.log(1-A))
    return L

def neuron_network(X, y, n1=32, learning_rate = 0.01, n_iter = 2000):
    n0 = X.shape[0]
    n2 = y.shape[0]
    params = init(n0, n1, n2)
    log_loss_tab = []
    for i in range(n_iter):
        activations = forward_propagation(X, params)
        log_loss_tab.append(log_loss(activations["A2"], y))
        gradients = back_propagation(X, y, activations, params)
        params = update(params, gradients, learning_rate)
        
    plt.plot(log_loss_tab)
    plt.show()
    return params


def predict(X, params):
    return forward_propagation(X, params)["A2"] >= 0.5


def predict_a_new_plante(x1, x2, W, b):
    new_plant = np.array([x1, x2])
    print(predict(new_plant, W, b))


params = neuron_network(X, y)

plt.scatter(X[0, :], X[1, :], c=y, cmap='summer')
plt.show()