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

def linear_forward(activation, weight, bias):
    weighted_sum = np.dot(weight, activation) + bias
    cache = (activation, weight, bias)
    return weighted_sum, cache


def sigmoid_(weighted_sum):
    return 1 / (1 + np.exp(-weighted_sum))


def relu_(weighted_sum):
    return weighted_sum * (weighted_sum > 0)


def drelu_(weighted_sum):
    return 1. * (weighted_sum > 0)


def dsigmoid_(weighted_sum):
    return sigmoid_(weighted_sum) * (1 - sigmoid_(weighted_sum))


def sigmoid(weighted_sum):
    return sigmoid_(weighted_sum), weighted_sum


def relu(weighted_sum):
    return relu_(weighted_sum), weighted_sum


def linear_activation_forward(activation_prev, weight, bias, activation):
    if activation == "sigmoid":
        weighted_sum, linear_cache = linear_forward(activation_prev, weight, bias)
        activation, activation_cache = sigmoid(weighted_sum)

    elif activation == "relu":
        weighted_sum, linear_cache = linear_forward(activation_prev, weight, bias)
        activation, activation_cache = relu(weighted_sum)

    cache = (linear_cache, activation_cache)

    return activation, cache

def layer_model_forward(x, parameters_):
    caches = []
    activation = x
    param_length = len(parameters_) // 2
    for l in range(1, param_length):
        activation_prev = activation
        activation, cache = linear_activation_forward(activation_prev, parameters_['W' + str(l)], parameters_['b' + str(l)], "relu")
        caches.append(cache)
    activation_linear, cache = linear_activation_forward(activation, parameters_['W' + str(param_length)], parameters_['b' + str(param_length)], "sigmoid")
    caches.append(cache)
    return activation_linear, caches


def compute_cost(activation_layer, y_):
    m = y_.shape[1]
    cost = -(1 / m) * np.sum((y_ * np.log(activation_layer) + (1 - y_) * np.log(1 - activation_layer)))
    cost = np.squeeze(cost)
    return cost

def linear_backward(deriv_weighted_sum, cache):
    activation_prev, weight, bias = cache
    m = activation_prev.shape[1]
    deriv_weight = (1 / m) * np.dot(deriv_weighted_sum, activation_prev.T)
    deriv_bias = (1 / m) * np.sum(deriv_weighted_sum, axis=1, keepdims=True)
    deriv_activation_prev = np.dot(weight.T, deriv_weighted_sum)
    return deriv_activation_prev, deriv_weight, deriv_bias


def relu_backward(deriv_activation, activation_cache):
    return deriv_activation * drelu_(activation_cache)


def sigmoid_backward(deriv_aactivation, activation_cache):
    return deriv_aactivation * dsigmoid_(activation_cache)


def linear_activation_backward(deriv_activation, cache, activation):
    linear_cache_, activation_cache = cache
    if activation == "relu":
        deriv_weighted_sum = relu_backward(deriv_activation, activation_cache)
        deriv_activation_prev, deriv_weight, deriv_bias = linear_backward(deriv_weighted_sum, linear_cache_)

    elif activation == "sigmoid":
        deriv_weighted_sum = sigmoid_backward(deriv_activation, activation_cache)
        deriv_activation_prev, deriv_weight, deriv_bias = linear_backward(deriv_weighted_sum, linear_cache_)
    return deriv_activation_prev, deriv_weight, deriv_bias


# back propogation for L layers
def L_model_backward(activation_layer, y_, caches):
    grads = {}
    cache_length = len(caches)
    deriv_activation_layer = - (np.divide(y_, activation_layer) - np.divide(1 - y_, 1 - activation_layer))

    current_cache = caches[cache_length - 1]
    grads["dA" + str(cache_length - 1)], grads["dW" + str(cache_length)], grads["db" + str(cache_length)] = linear_activation_backward(deriv_activation_layer,
                                                                                                      current_cache,
                                                                                                      "sigmoid")
    for l in reversed(range(cache_length - 1)):
        current_cache = caches[l]
        deriv_activation_prev_temp, deriv_weight_temp, deriv_bias_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache, "relu")
        grads["dA" + str(l)] = deriv_activation_prev_temp
        grads["dW" + str(l + 1)] = deriv_weight_temp
        grads["db" + str(l + 1)] = deriv_bias_temp
    return grads


# update parameters
def update_parameters(parameters_, grads, learning_rate):
    param_length = len(parameters_) // 2
    for l in range(param_length):
        parameters_["W" + str(l + 1)] = parameters_["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters_["b" + str(l + 1)] = parameters_["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]
    return parameters_

def layered_network_layer_model(x_, y_, layers_dims_, learning_rate=0.00042, num_iterations=3000, print_cost=False):
    np.random.seed(1)
    costs = []
    parameters_ = initialize_parameters_deep(layers_dims_)
    for i in range(0, num_iterations):
        AL, caches = layer_model_forward(x_, parameters_)
        cost = compute_cost(AL, y_)
        grads = L_model_backward(AL, y_, caches)
        parameters_ = update_parameters(parameters_, grads, learning_rate)
        if print_cost and i % 1000 == 0:
            print("Cost at iteration %i: %f" % (i, cost))
        if print_cost and i % 1000 == 0:
            costs.append(cost)

    return parameters_

def predict_layered_network_layer(x_, parameters_, results=False):
    a_l, caches = layer_model_forward(x_, parameters_)
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
parameters = layered_network_layer_model(X_train, Y_train_, layers_dims, num_iterations=15000, print_cost=True)
predictions_train_L = predict_layered_network_layer(X_train, parameters)
print(np.sum(predictions_train_L == y_train))

predictions_test_L = predict_layered_network_layer(X_test, parameters)
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
predicted_digit = predict_layered_network_layer(test_digit, parameters, True)
print('Predicted digit is : ' + str(predicted_digit))
