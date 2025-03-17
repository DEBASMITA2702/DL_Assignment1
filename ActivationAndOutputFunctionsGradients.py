import numpy as np
from ActivationAndOutputFunctions import softmax, sigmoid
import Utilities

def sigmoidGradient(input):
    '''
      Parameters:
        input : input for which to find the gradient of sigmoid function
      Returns:
        gradient of sigmoid value of the input
      Function:
        finds and returns the gradient of sigmoid value which is used during backpropagation
    '''

    # finding the sigmoid value of the input
    sigmoid_value = sigmoid(input)

    # performing gradient of sigmoid
    gradient = sigmoid_value * (1 - sigmoid_value)

    return gradient


def tanhGradient(input):
    '''
      Parameters:
        input : input for which to find the gradient of tanh function
      Returns:
        gradient of tanh value of the input
      Function:
        finds and returns the gradient of tanh value which is used during backpropagation
    '''

    # finding the tanh value of the input
    tanh_value = np.tanh(input)

    # performing gradient of tanh
    gradient = 1 - (tanh_value**2)

    return gradient


def reluGradient(input):
    '''
      Parameters:
        input : input for which to find the gradient of relu function
      Returns:
        gradient of relu value of the input
      Function:
        finds and returns the gradient of relu value which is used during backpropagation
    '''
    return 1 * (input > 0)


def identityGradient(input):
    '''
      Parameters:
        input : input for which to find the gradient of identity function
      Returns:
        gradient of identity value of the input
      Function:
        finds and returns the gradient of identity value which is used during backpropagation
    '''

    # finding the dimensions of the input (which is given as a batch)
    dim1 = input.shape[0]
    dim2 = input.shape[1]

    # creating a 2D array of all ones (because the gradient of an indentity fucntion is all ones)
    gradient = Utilities.create_ones(dim1, dim2)
    
    return gradient


def softmaxGradient(input):
    '''
      Parameters:
        input : input for which to find the gradient of softmax function
      Returns:
        gradient of softmax value of the input
      Function:
        finds and returns the gradient of softmax value which is used during backpropagation
    '''

    # finding the softmax value of the input
    softmax_value = softmax(input)

    # performing gradient of softmax
    gradient = softmax_value * (1 - softmax_value)

    return gradient