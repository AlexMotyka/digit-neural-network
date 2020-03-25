import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd

# stores iterations and costs for the Cost vs. Iterations graph
# first array stores iterations
# second array stores costs
cumulative_cost_data = [[],[]]

# initialize the weights and biases for the neural network
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
        if print_cost and i % 500 == 0:
            print("Cost at iteration %i: %f" % (i, cost))
            costs.append(cost)
            cumulative_cost_data[0].append(i)
            cumulative_cost_data[1].append(cost)

    return parameters_

def predict_layered_network_layer(x_, parameters_, results=False):
    a_l, caches = layer_model_forward(x_, parameters_)
    if results:
        return a_l.reshape(1, a_l.shape[0]), np.argmax(a_l, axis=0)
    prediction = np.argmax(a_l, axis=0)
    return prediction.reshape(1, prediction.shape[0])

train_data = []
correct_outputs = []
# arrays to store the test data
test_data = []
correct_class = []

# get iris data from data file
bank_data = pd.read_csv("bank.csv", delimiter=";")

split = np.random.rand(len(bank_data)) < 0.7
train = bank_data[split]
test = bank_data[~split]

for index, row in train.iterrows():
    train_data.append([row[0], row[1], row[2], row[3], row[4], row[5],row[6],row[7],row[8],row[9],row[10],row[11],row[12],row[13],row[14],row[15]])
    if row["y"] == "no":
        correct_outputs.append([0])
    else:
        correct_outputs.append([1])

for index, row in test.iterrows():
    test_data.append([row[0], row[1], row[2], row[3], row[4], row[5],row[6],row[7],row[8],row[9],row[10],row[11],row[12],row[13],row[14],row[15]])
    if row["y"] == "no":
        correct_class.append(0)
    else:
        correct_class.append(1)

# convert the lists to numpy array for performance, and then zip them together
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

num_output_neurons = 2

Y_train_ = np.zeros((num_output_neurons, y_train.shape[1]))
for i in range(y_train.shape[1]):
    Y_train_[y_train[0, i], i] = 1

Y_test_ = np.zeros((num_output_neurons, y_test.shape[1]))
for i in range(y_test.shape[1]):
    Y_test_[y_test[0, i], i] = 1

# the number of inputs
n_x = X_train.shape[0]

# number of hidden neurons
n_hidden = 50

# the number of output neurons
n_y = Y_train_.shape[0]
parameters = initialize_parameters_deep([n_x, n_hidden, n_y])
W1 = parameters["W1"]
b1 = parameters["b1"]

A = X_train
Z, linear_cache = linear_forward(A, W1, b1)

# N layer neural network
# 4 inputs, 10 outputs in the hidden layer, and 3 outputs in the last layer(1 for each iris class)
layers_dims = [n_x, n_hidden, n_y]
parameters = layered_network_layer_model(X_train, Y_train_, layers_dims, num_iterations=50000, print_cost=True)
predictions_train_L = predict_layered_network_layer(X_train, parameters)

# Creates a scatter plot that visualizes the cost over time
plt.plot(cumulative_cost_data[0], cumulative_cost_data[1], 'ro')
plt.grid(True)
plt.title('Cost vs Iterations during Training')
plt.xlabel('# of Iterations')
plt.ylabel('Cost')


print(np.sum(predictions_train_L == y_train))

predictions_test_L = predict_layered_network_layer(X_test, parameters)
print(np.sum(predictions_test_L == y_test))


# total correct predictions
total_correct = 0
# loop through the test data and feed each test digit through the network
index = 0
for point in test_data:
    # format the test digit as a valid numpy array
    test_point = np.asanyarray(point, dtype=np.uint8).reshape((n_x, 1)).T
    test_point = sc.transform(test_point).T
    predicted_class = predict_layered_network_layer(test_point, parameters, True)
    print("\n")
    # Output from the last layer of the network
    print(predicted_class[0])
    # The network prediction is the digit with the highest confidence
    prediction = np.argmax(predicted_class[0])
    print("Prediction is: " + str(prediction) + " and Actual is: " + str(correct_class[index]))
    if prediction == correct_class[index]:
        total_correct += 1
        print("CORRECT")
    else:
        print("WRONG")
    index += 1
print("\nTotal correct: " + str(total_correct) + "/30")

plt.show()
