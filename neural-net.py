import numpy as np
import digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def initialize_parameters_deep(layer_dims):
    np.random.seed(3)
    parameters_ = {}
    layer_dim_len = len(layer_dims)
    for l in range(1, layer_dim_len):
        parameters_['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
        parameters_['b' + str(l)] = np.zeros((layer_dims[l], 1))
    return parameters_

def linear_forward(a, w, b):
    z = np.dot(w, a) + b
    cache = (a, w, b)
    return z, cache


def sigmoid_(z):
    return 1 / (1 + np.exp(-z))


def relu_(z):
    return z * (z > 0)


def drelu_(z):
    return 1. * (z > 0)


def dsigmoid_(z):
    return sigmoid_(z) * (1 - sigmoid_(z))


def sigmoid(z):
    return sigmoid_(z), z


def relu(z):
    return relu_(z), z


def linear_activation_forward(a_prev, w, b, activation):
    if activation == "sigmoid":
        z, linear_cache = linear_forward(a_prev, w, b)
        a, activation_cache = sigmoid(z)

    elif activation == "relu":
        z, linear_cache = linear_forward(a_prev, w, b)
        a, activation_cache = relu(z)

    cache = (linear_cache, activation_cache)

    return a, cache

def L_model_forward(x, parameters_):
    caches = []
    A = x
    L = len(parameters_) // 2
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters_['W' + str(l)], parameters_['b' + str(l)], "relu")
        caches.append(cache)
    AL, cache = linear_activation_forward(A, parameters_['W' + str(L)], parameters_['b' + str(L)], "sigmoid")
    caches.append(cache)
    return AL, caches


def compute_cost(a_l, y):
    m = y.shape[1]
    cost = -(1 / m) * np.sum((y * np.log(a_l) + (1 - y) * np.log(1 - a_l)))
    cost = np.squeeze(cost)
    return cost

def linear_backward(d_z, cache):
    a_prev, w, b = cache
    m = a_prev.shape[1]
    d_w = (1 / m) * np.dot(d_z, a_prev.T)
    db = (1 / m) * np.sum(d_z, axis=1, keepdims=True)
    d_a_prev = np.dot(w.T, d_z)
    return d_a_prev, d_w, db


def relu_backward(d_a, activation_cache):
    return d_a * drelu_(activation_cache)


def sigmoid_backward(d_a, activation_cache):
    return d_a * dsigmoid_(activation_cache)


def linear_activation_backward(d_a, cache, activation):
    linear_cache_, activation_cache = cache
    if activation == "relu":
        d_z = relu_backward(d_a, activation_cache)
        d_a_prev, d_w, db = linear_backward(d_z, linear_cache_)

    elif activation == "sigmoid":
        d_z = sigmoid_backward(d_a, activation_cache)
        d_a_prev, d_w, db = linear_backward(d_z, linear_cache_)
    return d_a_prev, d_w, db


# back propogation for L layers
def L_model_backward(a_l, y, caches):
    grads = {}
    cache_length = len(caches)
    deriv_a_l = - (np.divide(y, a_l) - np.divide(1 - y, 1 - a_l))

    current_cache = caches[cache_length - 1]
    grads["dA" + str(cache_length - 1)], grads["dW" + str(cache_length)], grads["db" + str(cache_length)] = linear_activation_backward(deriv_a_l,
                                                                                                      current_cache,
                                                                                                      "sigmoid")
    for l in reversed(range(cache_length - 1)):
        current_cache = caches[l]
        deriv_a_prev_temp, deriv_w_temp, deriv_b_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache, "relu")
        grads["dA" + str(l)] = deriv_a_prev_temp
        grads["dW" + str(l + 1)] = deriv_w_temp
        grads["db" + str(l + 1)] = deriv_b_temp
    return grads


# update parameters
def update_parameters(parameters_, grads, learning_rate):
    param_length = len(parameters_) // 2
    for l in range(param_length):
        parameters_["W" + str(l + 1)] = parameters_["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters_["b" + str(l + 1)] = parameters_["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]
    return parameters_

def L_layer_model(x, y, layers_dims_, learning_rate=0.00042, num_iterations=3000, print_cost=False):
    np.random.seed(1)
    costs = []
    parameters_ = initialize_parameters_deep(layers_dims_)
    for i in range(0, num_iterations):
        AL, caches = L_model_forward(x, parameters_)
        cost = compute_cost(AL, y)
        grads = L_model_backward(AL, y, caches)
        parameters_ = update_parameters(parameters_, grads, learning_rate)
        if print_cost and i % 1000 == 0:
            print("Cost after iteration %i: %f" % (i, cost))
        if print_cost and i % 1000 == 0:
            costs.append(cost)

    return parameters_

def predict_L_layer(x, parameters_, results=False):
    a_l, caches = L_model_forward(x, parameters_)
    if results:
        return a_l.reshape(1, a_l.shape[0]), np.argmax(a_l, axis=0)
    prediction = np.argmax(a_l, axis=0)
    return prediction.reshape(1, prediction.shape[0])


train_data = []
correct_outputs = []

for digit in digits.zeros:
    train_data.append(digit)
    correct_outputs.append([0])
for digit in digits.ones:
    train_data.append(digit)
    correct_outputs.append([1])
for digit in digits.twos:
    train_data.append(digit)
    correct_outputs.append([2])
for digit in digits.threes:
    train_data.append(digit)
    correct_outputs.append([3])
for digit in digits.fours:
    train_data.append(digit)
    correct_outputs.append([4])
for digit in digits.fives:
    train_data.append(digit)
    correct_outputs.append([5])
for digit in digits.sixes:
    train_data.append(digit)
    correct_outputs.append([6])
for digit in digits.sevens:
    train_data.append(digit)
    correct_outputs.append([7])
for digit in digits.eights:
    train_data.append(digit)
    correct_outputs.append([8])
for digit in digits.nines:
    train_data.append(digit)
    correct_outputs.append([9])

train_data = np.asanyarray(train_data, dtype=np.uint8)
correct_outputs = np.asanyarray(correct_outputs, dtype=np.uint8)
images_and_labels = list(zip(train_data, correct_outputs))

# Define variables
n_samples = len(train_data)
print(n_samples)

x = train_data.reshape((n_samples, -1))
print(x.shape)

y = correct_outputs
print(y.shape)



X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Feature Scaling


sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train = X_train.T
X_test = X_test.T
y_train = y_train.reshape(y_train.shape[0], 1)
y_test = y_test.reshape(y_test.shape[0], 1)
y_train = y_train.T
y_test = y_test.T

Y_train_ = np.zeros((10, y_train.shape[1]))
for i in range(y_train.shape[1]):
    Y_train_[y_train[0, i], i] = 1

Y_test_ = np.zeros((10, y_test.shape[1]))
for i in range(y_test.shape[1]):
    Y_test_[y_test[0, i], i] = 1

n_x = X_train.shape[0]
n_h = 10
n_y = Y_train_.shape[0]
parameters = initialize_parameters_deep([n_x, 10, n_y])
W1 = parameters["W1"]
b1 = parameters["b1"]

A = X_train
Z, linear_cache = linear_forward(A, W1, b1)

# N layer neural network
layers_dims = [n_x, 60, n_y]
parameters = L_layer_model(X_train, Y_train_, layers_dims, num_iterations=3000, print_cost=True)
predictions_train_L = predict_L_layer(X_train, parameters)
print(np.sum(predictions_train_L == y_train))

predictions_test_L = predict_L_layer(X_test, parameters)
print(np.sum(predictions_test_L == y_test))

test_digit = np.asanyarray([0,0,1,0,0,
            0,1,1,0,0,
            0,0,1,0,0,
            0,0,1,0,0,
            0,0,1,0,0,
            0,0,1,0,0,
            0,0,1,0,0,
            0,0,1,0,0,
            0,0,1,0,0], dtype=np.uint8).reshape((45, 1)).T
test_digit = sc.transform(test_digit).T
predicted_digit = predict_L_layer(test_digit, parameters, True)
print('Predicted digit is : ' + str(predicted_digit))



