# Digit Neural-Network

This is a neural network that solves the classic problem of identifying the digits 0-9. In this problem digits are represented by a 1-dimensional array of 1's and 0's, where the 1's represent dark pixels and the 0's represent blank pixels. An example of this would be the digit 0 which could be represented by the following 1-dimensional array:

[0,**1,1,1**,0,  
**1**,0,0,0,**1**,  
 **1**,0,0,0,**1**,  
 **1**,0,0,0,**1**,  
 **1**,0,0,0,**1**,  
 **1**,0,0,0,**1**,  
 **1**,0,0,0,**1**,  
 **1**,0,0,0,**1**,  
 0,**1,1,1**,0]  

Our Artificial Neural Network has an input layer with 45 inputs, a hidden layer with 60 outputs, and an output layer with 10 outputs. The 10 final outputs represent the 10 digits in order from 0 to 9. Each output is bound between 0 and 1 and they represent the confidence score that the digit they represent is the correct digit classification. For example, the example output array below represents a 90% confidence that the input to the neural network is the digit 6.

```
[0.1, 0.2, 0.1, 0.0, 0.0, 0.1, 0.9, 0.1, 0.0, 0.1]
```

The weights and biases for each layer are randomly initialized when the neural network is created. In the first layer, there are 2700 total weights (60 weights for each of the 45 inputs) and 60 biases (1 for each of the 10 outputs in the hidden layer). The second layer has 600 weights (10 for each of the 60 inputs coming from the hidden layer) and 10 biases (1 for each of the 10 final outputs).

### Activation Function

For the hidden layer, we used the Relu (Rectified Linear Unit) function as the activation function. The output layer uses the sigmoid function to give a probability outcome that is always in the range of 0 to 1, which is ideal as we want the output to represent the neural net’s confidence in its classification of a digit.


### Loss function & Backward Propagation

For our loss function, we implemented the Sum of Squares Error as per the requirements of the mini-project. Gradient Descent was used to determine the changes applied to the weights and biases in the neural network.

### Data Generation

To train our neural network we created a script to generate digits with small mutations (a flipped pixel). These random mutations slightly alter the appearance of the digits, but the general structure of the digit remains. For our training set, we used 100 versions of each digit.

## Results

### Cost vs. Iterations

After testing various iteration sizes we found that 35000 iterations produced the most accurate predictions. A plot of the cost mapped against the iterations can be seen below. Initially, the cost function value decreases rapidly however its rate of decrease slows down as the iterations increase.

### Prediction Accuracy

To test our neural network we fed forward three variations of each digit for a total of 30 tests. With 35000 iterations during the training phase, our network was able to predict the test digits with an accuracy of 90% (27/30 correct). To choose the predicted digit the network selects the index of the output array with the highest confidence. The network had difficulties predicting eights (66% accuracy), sevens (66% accuracy), and fives (66% accuracy). In these instances of failure the network thought an eight was a nine, a seven was a one, and a five was a nine. These failures are all reasonable as the problem space of 5x9 pixels is very small and a mutation in the correct place can make any of these digits look similar to the other. 

An interesting thing to note is our findings on prediction accuracy versus prediction confidence. During testing with different iteration sizes, we found that as our iterations increased to 35000 the neural net’s prediction accuracy increased, but the confidence it had in its predictions for similar digits (such as 0 and 8, or 1 and 7) decreased. This could be due to the fact that such similar digits trend toward borderline predictions with very small differences in cases where a mutation in a digit causes it to appear like another digit.


## Running on Linux/Ubuntu

Git clone the repo and make sure you have python, pip, and virtualenv installed. Then execute the following in your terminal:

```
cd digit-neural-network
```
```
virtualenv env
```
```
source env/bin/activate
```
```
pip install -r requirements.txt
```
```
python neural-net.py
```

## Running on Windows

Git clone the repo and make sure you have python, pip, and virtualenv installed. Then execute the following in your terminal:

```
cd digit-neural-network
```
```
virtualenv env
```
```
.\env\Scripts\activate.bat
```
```
pip install -r requirements.txt
```
```
python neural-net.py
```

number_gen.py is a helper script that generates variations of each digit using mutations(flipping a pixel value). These generated digits are used for training the network.
