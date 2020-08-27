# Digit Neural-Network

This is a neural network that solves the classic problem of identifying the digits 0-9. In this problem digits are represented by a 1-dimensional array of 1's and 0's, where the 1's represent dark pixels and the 0's represent blank pixels. An example of this would be the digit 0 which could be represented by the following 1-dimensional array:

[0,**1,1,1** ,0,  
**1**,0,0,0,**1**,  
 **1**,0,0,0,**1**,  
 **1**,0,0,0,**1**,  
 **1**,0,0,0,**1**,  
 **1**,0,0,0,**1**,  
 **1**,0,0,0,**1**,  
 **1**,0,0,0,**1**,  
 0,**1,1,1**,0]  

To run the neural network 'cd' into the project directory and run 'python neural-net.py'

neural-net.py is the script that trains and tests the neural network. It leverages data from digits.py and testDigits.py

number_gen.py is a helper script that generates variations of each digit using mutations(flipping a pixel value). These generated digits are used for training the network
