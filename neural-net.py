import numpy as np

def sigmoid(z):
    return 1.0/(1+ np.exp(-z))

def sigmoid_derivative(z):
    return z * (1.0 -z)



main():

    #### Training data
    # matrix of inputs to train the neural net
    train_data = np.array([])
    # matrix of correct outputs for the above data
    correct_outputs = np.array([])
    ####

    # matrix of inputs to test the neural net
    test_data = np.array([])



if __name__ == "__main__":
    main()
