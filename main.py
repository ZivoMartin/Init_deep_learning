import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_circles

X, y = make_circles(n_samples=100, noise=0.1, factor=0.3, random_state = 0)
X = X.T
y = y.reshape((1, y.shape[0]))



def init(dim_tab):
    params = {}
    i = 1
    size = len(dim_tab)

    while(i < size):
        params["W"+str(i)] = np.random.randn(dim_tab[i], dim_tab[i-1])
        params["b"+str(i)] = np.random.randn(dim_tab[i], 1) 
        i += 1
    
    return params


def forward_propagation(X, params):
    activations = {"A0": X}
    size = len(params) // 2
    i = 1
    while(i <= size):
        i_str = str(i)
        Zi = params["W"+i_str].dot(activations["A"+str(i-1)]) + params["b"+i_str]
        activations["A"+i_str] = 1 / (1 + np.exp(-Zi))
        i += 1
    
    return activations
    

def back_propagation(X, y, activations, params):
    m = y.shape[1]
    gradients = {}
    i = len(params) // 2
    dZ = activations["A"+str(i)] - y

    while(i >= 1):
        gradients["dW"+str(i)] = 1/m * dZ.dot(activations["A"+str(i-1)].T)
        gradients["db"+str(i)] = 1/m * np.sum(dZ, axis=1, keepdims = True)
        if(i > 1):
            dZ = np.dot(params["W"+str(i)].T, dZ) * activations["A"+str(i-1)]*(1-activations["A"+str(i-1)])
        i -= 1
    
    return gradients
 
def update(params, gradients, learning_rate):
    i = 1
    size = len(params) // 2
    while(i <= size):
        params["W"+str(i)] = params["W"+str(i)] - learning_rate * gradients["dW"+str(i)]
        params["b"+str(i)] = params["b"+str(i)] - learning_rate * gradients["db"+str(i)]
        i += 1
    return params

def log_loss(A, y):
    L = 1/len(y) * np.sum(-y*np.log(A) - (1-y) * np.log(1-A))
    return L

def neuron_network(X, y, dim_tab, learning_rate = 0.1, n_iter = 1000):
    params = init(dim_tab)
    log_loss_tab = []
    couche_finale = str(len(params) // 2)
    for i in range(n_iter):
        activations = forward_propagation(X, params)
        log_loss_tab.append(log_loss(activations["A"+couche_finale], y))
        gradients = back_propagation(X, y, activations, params)
        params = update(params, gradients, learning_rate)
        
    
    return params, log_loss_tab


def predict(X, params, last_couche):
    return forward_propagation(X, params)["A"+str(last_couche)] >= 0.5



dim_tab = [X.shape[0], 32, 32, 32, y.shape[0]]
params, log_loss_tab = neuron_network(X, y, dim_tab)

# plt.plot(log_loss_tab)
# plt.show()
fig, ax = plt.subplots()
ax.scatter(X[0, :], X[1, :], c=y, cmap="summer", s=50)
x0_lim = ax.get_xlim()
x1_lim = ax.get_ylim()
resolution = 100
x0 = np.linspace(x0_lim[0], x0_lim[1], resolution)
x1 = np.linspace(x1_lim[0], x1_lim[1], resolution)
X0, X1 = np.meshgrid(x0, x1)
XX = np.vstack((X0.ravel(), X1.ravel()))
Z = predict(XX, params, len(dim_tab)-1)
Z = Z.reshape((resolution, resolution))
ax.pcolormesh(X0, X1, Z, cmap="summer", alpha=0.3, zorder=-1)
ax.contour(X0, X1, Z, colors="green")
plt.show()