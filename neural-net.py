import numpy as np
import digits


class NeuralNet:
    def __init__(self, train_data, correct_outputs):
        # matrix of inputs for training
        self.train_data = train_data
        # matrix of correct outputs
        self.correct_outputs = correct_outputs

        ### Generate random weights
        self.weights_1 = []
        self.weights_2 = []

    # feed the inputs through the layers
    def feed_forward(self, test_data = None):
        # use test_data if it exists
        if test_data:
            # feed forward using test test_data
            pass
        else:
            # feed forward using train_data
            pass

    def backpropogation():
        # update the weights using Sum Squared Error
        # weights_2_change = sum squared error
        # weights_1_change = sum squared error

        # self.weights_1 += weights_1_change
        # self.weights_2 += weights_2_change
        pass



def sigmoid(z):
    return 1.0/(1+ np.exp(-z))

def sigmoid_derivative(z):
    return z * (1.0 -z)

def main():
    print(digits.one)
    #### Training data
    # matrix of digit inputs to train the neural net
    train_data = np.array([ digits.zero,
                            digits.one,
                            digits.two,
                            digits.three,
                            digits.four,
                            digits.five,
                            digits.six,
                            digits.seven,
                            digits.eight,
                            digits.nine ])
    # matrix of correct neural net outputs for the above data
    correct_outputs = np.array(
                                [1,0,0,0,0,0,0,0,0,0],
                                [0,1,0,0,0,0,0,0,0,0],
                                [0,0,1,0,0,0,0,0,0,0],
                                [0,0,0,1,0,0,0,0,0,0],
                                [0,0,0,0,1,0,0,0,0,0],
                                [0,0,0,0,0,1,0,0,0,0],
                                [0,0,0,0,0,0,1,0,0,0],
                                [0,0,0,0,0,0,0,1,0,0],
                                [0,0,0,0,0,0,0,0,1,0],
                                [0,0,0,0,0,0,0,0,0,1], )
    ####

    # matrix of inputs to test the neural net
    test_data = np.array([])



if __name__ == "__main__":
    main()
