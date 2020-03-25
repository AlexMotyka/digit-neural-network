import numpy as np
import digits
import testDigits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# stores iterations and costs for the Cost vs. Iterations graph
# first array stores iterations
# second array stores costs
cumulative_cost_data = [[],[]]

train_data = []
correct_outputs = []
"""----------Activation Functions----------"""
def relu_backward(deriv_activation, activation_cache):
    return deriv_activation * 1. * (activation_cache> 0)


def sigmoid_backward(deriv_activation, activation_cache):
    return deriv_activation * (1 / (1 + np.exp(-activation_cache))) * (1 - (1 / (1 + np.exp(-activation_cache))))


def sigmoid(weighted_sum):
    return (1 / (1 + np.exp(-weighted_sum))), weighted_sum


def relu(weighted_sum):
    return weighted_sum * (weighted_sum > 0), weighted_sum
"""----------------------------------------"""

"""
Init the parameters
"""
def initialize_parameters_deep(layer_dims):
    np.random.seed(3)
    parameters_ = {}
    layer_dim_len = len(layer_dims)
    for l in range(1, layer_dim_len):
        parameters_['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
        parameters_['b' + str(l)] = np.zeros((layer_dims[l], 1))
    return parameters_

"""
linear_forward
*no activation*
"""
def linear_forward(activation, weight, bias):
    weighted_sum = np.dot(weight, activation) + bias
    cache = (activation, weight, bias)
    return weighted_sum, cache

"""
linear_activation_forward
This function is a linear_forward *with* activation in the network
activation_prev - the previous network output
activation - the output of the network
"""
def linear_activation_forward(activation_prev, weight, bias, activation):
    if activation == "sigmoid":
        weighted_sum, linear_cache_ = linear_forward(activation_prev, weight, bias)
        activation, activation_cache = sigmoid(weighted_sum)

    elif activation == "relu":
        weighted_sum, linear_cache_ = linear_forward(activation_prev, weight, bias)
        activation, activation_cache = relu(weighted_sum)

    cache = (linear_cache_, activation_cache)

    return activation, cache

"""
forward propagation
- move forward for all hidden layers, then one final output layer
- hidden layers are relu activation
- output layer is sigmoid
"""
def layer_model_forward(x_, parameters_):
    caches = []
    activation = x_
    param_length = len(parameters_) // 2
    for l in range(1, param_length):
        activation_prev = activation
        activation, cache = linear_activation_forward(activation_prev, parameters_['W' + str(l)], parameters_['b' + str(l)], "relu")
        caches.append(cache)
    activation_linear, cache = linear_activation_forward(activation, parameters_['W' + str(param_length)], parameters_['b' + str(param_length)], "sigmoid")
    caches.append(cache)
    return activation_linear, caches

"""
compute_cost
This function calculates cost with the Sum of Squares Deviation formula
y_ - the true output
activation_layer - is the output of the network (the value)
sse/ssd - the sum of squared deviations/error between the two 
"""
def compute_cost(activation_layer, y_):
    true_output_std = np.std(y_)
    network_output_std = np.std(activation_layer)
    sse = np.sum(np.square(true_output_std - network_output_std)) * 100.0
    return sse

def linear_backward(deriv_weighted_sum, cache):
    activation_prev, weight, bias = cache
    m = activation_prev.shape[1]
    deriv_weight = (1 / m) * np.dot(deriv_weighted_sum, activation_prev.T)
    deriv_bias = (1 / m) * np.sum(deriv_weighted_sum, axis=1, keepdims=True)
    deriv_activation_prev = np.dot(weight.T, deriv_weighted_sum)
    return deriv_activation_prev, deriv_weight, deriv_bias

"""
linear backwards uses the derivatives of the activation functions instead
"""
def linear_activation_backward(deriv_activation, cache, activation):
    linear_cache_, activation_cache = cache

    # if activation was on a hidden layer, do relu activation
    if activation == "relu":
        deriv_weighted_sum = relu_backward(deriv_activation, activation_cache)
        deriv_activation_prev, deriv_weight, deriv_bias = linear_backward(deriv_weighted_sum, linear_cache_)

    # if the activation is on output layer, do sigmoid
    elif activation == "sigmoid":
        deriv_weighted_sum = sigmoid_backward(deriv_activation, activation_cache)
        deriv_activation_prev, deriv_weight, deriv_bias = linear_backward(deriv_weighted_sum, linear_cache_)
    return deriv_activation_prev, deriv_weight, deriv_bias


# back propogation
def layered_model_backward(activation_layer, y_, caches):
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


# update the parameters
def update_parameters(parameters_, grads, learning_rate):
    param_length = len(parameters_) // 2
    for l in range(param_length):
        parameters_["W" + str(l + 1)] = parameters_["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters_["b" + str(l + 1)] = parameters_["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]
    return parameters_

def layered_network_layer_model(x_, y_, layers_dims_, learning_rate=0.00042, num_iterations=3000, print_cost=False):
    np.random.seed(1314)
    costs = []
    parameters_ = initialize_parameters_deep(layers_dims_)
    for j in range(0, num_iterations):
        activation_layer, caches = layer_model_forward(x_, parameters_)
        sse = compute_cost(activation_layer, y_)
        grads = layered_model_backward(activation_layer, y_, caches)
        parameters_ = update_parameters(parameters_, grads, learning_rate)
        if print_cost and j % 500 == 0:
            print("Cost at iteration %i: %f SSE" % (j, sse))
            costs.append(sse)
            cumulative_cost_data[0].append(j)
            cumulative_cost_data[1].append(sse)

    return parameters_

def predict_layered_network_layer(x_, parameters_, results=False):
    a_l, caches = layer_model_forward(x_, parameters_)
    if results:
        return a_l.reshape(1, a_l.shape[0]), np.argmax(a_l, axis=0)
    prediction_ = np.argmax(a_l, axis=0)
    return prediction_.reshape(1, prediction_.shape[0]), prediction_

'''----- main code starts here ------'''
# read in data from digits.py and append it to the training data
# the correct output the network should produce for that digit is appended to the correct_outputs array 
# for digit in digits.zeros:
#     train_data.append(digit)
#     correct_outputs.append([0])
# for digit in digits.ones:
#     train_data.append(digit)
#     correct_outputs.append([1])
# for digit in digits.twos:
#     train_data.append(digit)
#     correct_outputs.append([2])
# for digit in digits.threes:
#     train_data.append(digit)
#     correct_outputs.append([3])
# for digit in digits.fours:
#     train_data.append(digit)
#     correct_outputs.append([4])
# for digit in digits.fives:
#     train_data.append(digit)
#     correct_outputs.append([5])
# for digit in digits.sixes:
#     train_data.append(digit)
#     correct_outputs.append([6])
# for digit in digits.sevens:
#     train_data.append(digit)
#     correct_outputs.append([7])
# for digit in digits.eights:
#     train_data.append(digit)
#     correct_outputs.append([8])
# for digit in digits.nines:
#     train_data.append(digit)
#     correct_outputs.append([9])

with open("./datasets/breast_cancer_data/wdbc.data", "r") as f:
    lines = [line.rstrip('\n') for line in f]


answers_arr = []
input_arr = []

for line in lines:
    line = line.split(',')
    if line[1] == 'M':
        answers_arr.append(1)
    else:
        answers_arr.append(0)
    line.remove(line[0])
    line.remove(line[0])
    input_arr.append(line)



# convert the lists to numpy array for performance, and then zip them together
train_data = np.asanyarray(input_arr)
correct_outputs = np.asanyarray(answers_arr)
data_and_labels = list(zip(train_data, correct_outputs))

# Define variables
n_samples = len(train_data)
print("number of samples: " + str(n_samples))

x = train_data.reshape((n_samples, -1))
print("train data shape: " + str(x.shape))

y = correct_outputs
print("answers data shape: " + str(y.shape))

# split the set and use 30 percent as test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
# # scale the features
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

x_train = x_train.T
x_test = x_test.T
y_train = y_train.reshape(y_train.shape[0], 1)
y_test = y_test.reshape(y_test.shape[0], 1)
y_train = y_train.T
y_test = y_test.T

# generate weight matrix
Y_train_ = np.zeros((10, y_train.shape[1]))
for i in range(y_train.shape[1]):
    Y_train_[y_train[0, i], i] = 1

Y_test_ = np.zeros((10, y_test.shape[1]))
for i in range(y_test.shape[1]):
    Y_test_[y_test[0, i], i] = 1

n_x = x_train.shape[0]
n_h = 10
n_y = Y_train_.shape[0]
parameters = initialize_parameters_deep([n_x, 10, n_y])
W1 = parameters["W1"]
b1 = parameters["b1"]

a = x_train
weighted_sum_, linear_cache = linear_forward(a, W1, b1)

# N layer neural network
# 45 inputs, 60 outputs in the hidden layer, and 10 outputs in the last layer(1 for each digit)
layers_dims = [n_x, 60, n_y]
parameters = layered_network_layer_model(x_train, Y_train_, layers_dims, num_iterations=35000, print_cost=True)
predictions_train_L, raw_prediction_list_train = predict_layered_network_layer(x_train, parameters)

# Creates a scatter plot that visualizes the cost over time
plt.plot(cumulative_cost_data[0], cumulative_cost_data[1], 'ro')
plt.grid(True)
plt.title('Cost vs Iterations during Training')
plt.xlabel('# of Iterations')
plt.ylabel('Cost')
plt.show()


predictions_test_L, raw_prediction_list_test = predict_layered_network_layer(x_test, parameters)

expected_results = y_test[0].tolist()
predicted_results = list(np.array(raw_prediction_list_test))

# calculate the confusion matrix
actual_y_predict_y = 0
actual_n_predict_y = 0
actual_y_predict_n = 0
actual_n_predict_n = 0
count = 0
for value in predicted_results:
    temp_val = value.astype(np.int32)
    if temp_val == 1 and expected_results[count] == 1:
        actual_y_predict_y = actual_y_predict_y + 1
    elif temp_val  == 0 and expected_results[count] == 0:
        actual_n_predict_n = actual_n_predict_n + 1
    elif temp_val  == 0 and expected_results[count] == 1:
        actual_y_predict_n = actual_y_predict_n + 1
    elif temp_val  == 1 and expected_results[count] == 0:
        actual_n_predict_y = actual_n_predict_y + 1
    count = count + 1

print("A YES : P YES - " + str(actual_y_predict_y) + '\n')
print("A YES : P NO - " + str(actual_y_predict_n) + '\n')
print("A NO : P NO - " + str(actual_n_predict_n) + '\n')
print("A NO : P YES - " + str(actual_n_predict_y) + '\n')
