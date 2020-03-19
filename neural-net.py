import numpy as np
import digits
import testDigits


class NeuralNet:
    def __init__(self, train_data, correct_outputs):
        # matrix of inputs for training
        self.train_data = train_data
        # matrix of correct outputs
        self.correct_outputs = correct_outputs

        ### Generate random weights
        # matrix of 5 weights for each of the 45 inputs
        self.weights_1 = np.random.rand(self.train_data.shape[1],5)
        # matrix of 1 weight for each of the 5 inputs
        self.weights_2 = np.random.rand(5, 1)

    # feed the inputs through the layers
    def feed_forward(self, test_data = np.array([])):
        # use train_data
        if test_data.size == 0:
            # feed input through first layer
            self.layer_1_out = sigmoid(np.dot(self.train_data, self.weights_1))
            # feed layer_1_out through second layer to get the network outputs
            self.train_outputs = sigmoid(np.dot(self.layer_1_out, self.weights_2))
        # use test_data
        else:
            # feed input through first layer
            self.layer_1_out = sigmoid(np.dot(test_data, self.weights_1))
            # feed layer_1_out through second layer to get the network outputs
            prediction = sigmoid(np.dot(self.layer_1_out, self.weights_2))
            return prediction


    def backpropogation(self):
        # print(self.train_outputs.shape)
        """
        The loss function for backpropogation is the sum-squared error.
        To determine how we should update the weights the derivative of the
        sum-squared error is used to find the gradient.
        This gradient is added to our weights matrix and the weights are updated
        """
        # update the weights using Sum Squared Error as the loss function
        # Use the derivative of the loss function to determine
        weights_2_delta = np.dot(self.layer_1_out.T, self.sum_squared_derivative())
        weights_1_delta= np.dot(self.train_data.T, (np.dot(self.sum_squared_derivative(), self.weights_2.T) * sigmoid_derivative(self.layer_1_out)))

        self.weights_1 += weights_1_delta
        self.weights_2 += weights_2_delta

    def sum_squared_derivative(self):
        return 2*(self.correct_outputs - self.train_outputs) * sigmoid_derivative(self.train_outputs)


def sigmoid(z):
    return 1.0/(1+ np.exp(-z))

def sigmoid_derivative(z):
    return z * (1.0 - z)

def main():
    #### Training data
    # matrix of digit inputs to train the neural net
    train_data = np.array([ digits.zero0,
                            digits.zero1,
                            digits.zero2,
                            digits.zero3,
                            digits.zero4,
                            digits.one,
                            digits.two,
                            digits.three,
                            digits.four,
                            digits.five,
                            digits.six,
                            digits.seven,
                            digits.eight,
                            digits.nine ])
    #
    # # matrix of correct neural net outputs for the above data
    correct_outputs = np.array([[1],
                                [1],
                                [1],
                                [1],
                                [1],
                                [0],
                                [0],
                                [0],
                                [0],
                                [0],
                                [0],
                                [0],
                                [0],
                                [0]])
    ####
    # train_data = []
    # correct_outputs = []

    # for digit in digits.zeros:
    #     train_data.append(digit)
    #     correct_outputs.append([1])

    # non_zeros = [digits.one,
    #              digits.two,
    #              digits.three,
    #              digits.four,
    #              digits.five,
    #              digits.six,
    #              digits.seven,
    #              digits.eight,
    #              digits.nine ]
    # for digit in non_zeros:
    #     train_data.append(digit)
    #     correct_outputs.append([0])

    # train_data = np.asarray(train_data, dtype=np.float64)

    # correct_outputs = np.asarray(correct_outputs, dtype=np.float64)

    # matrix of inputs to test the neural net
    test_zero = np.array([testDigits.test_zero1])

    test_one = np.array([digits.one])

    test_two = np.array([digits.two])

    network = NeuralNet( train_data, correct_outputs)

    for iteration in range(15000):
        network.feed_forward()
        network.backpropogation()

    #print (network.train_outputs)

    print(network.feed_forward(test_zero))
    print(network.feed_forward(test_one))
    print(network.feed_forward(test_two))




if __name__ == "__main__":
    main()
