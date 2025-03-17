import numpy as np

'''The following are the utility functions used across various classes and functions across the whole assignment'''

def getShape0(npArray):
    '''
      Parameters:
        npArray : A numpy array for which to find the dimension
      Returns:
        An integer denoting the shape of the array in 1st dimension
      Function:
        Finds and return the shape of the 1st dimension of an numpy array
    '''
    return npArray.shape[0]


'''generates and returns a numpy array of random numbers in a range'''
def generateRandom(range1, range2, replaceParam):
    '''
      Parameters:
        range1 : starting point of the range
        range2 : ending point of the range
        replaceParam : boolean representing whether to replace the values of not
      Returns:
        A numpy array
      Function:
        Creates a numpy array of random values picked in the given range
    '''
    return np.random.choice(range1, range2, replace = replaceParam)


def deleteData(sourceData, deleteData):
    '''
      Parameters:
        sourceData : the original data where the deletion will happen
        deleteData : the data which will be deleted 
      Returns:
        A numpy array
      Function:
        deletes some given data from another data in a numpy array
    '''
    return np.delete(sourceData, deleteData, axis = 0)


def random1DFloat(dim1):
    '''
      Parameters:
        dim1 : dimension value (number of columns) for the array
      Returns:
        A numpy array
      Function:
        creates a random 1D vector of given dimension with the data type as float64 
    '''
    return np.random.randn(dim1).astype(np.float64)


def random2D(dim1, dim2):
    '''
      Parameters:
        dim1 : dimension value (number of rows) for the array
        dim2 : dimension value (number of columns) for the array
      Returns:
        A numpy array
      Function:
        creates a random 2D vector of given dimensions
    '''
    return np.random.randn(dim1, dim2)


def random2DFloat(dim1, dim2):
    '''
      Parameters:
        dim1 : dimension value (number of rows) for the array
        dim2 : dimension value (number of columns) for the array
      Returns:
        A numpy array
      Function:
        creates a random 2D vector of given dimension with the data type as float64
    '''
    return np.random.randn(dim1, dim2).astype(np.float64)


def xav_nor(input, output):
    '''
      Parameters:
        input : number of neuron connections coming into the layer
        output : number of neuron connections going from the layer
      Returns:
        A float64 which represents the xavier normal factor
      Function:
        finds the xavier normal factor with the data type as float64 with the given number of input and output connections
    '''
    return np.sqrt(2 / (input + output)).astype(np.float64)


def xav_uni(input, output):
    '''
      Parameters:
        input : number of neuron connections coming into the layer
        output : number of neuron connections going from the layer
      Returns:
        A float64 which represents the xavier uniform factor
      Function:
        finds the xavier uniform factor with the data type as float64 with the given number of input and output connections
    '''
    return np.sqrt(6 / (input + output)).astype(np.float64)


def he_nor(input):
    '''
      Parameters:
        input : number of neuron connections coming into the layer
      Returns:
        A float64 which represents the he normal factor
      Function:
        finds the he normal factor with the data type as float64 with the given number of input connections
    '''
    return np.sqrt(2 / input).astype(np.float64)


def he_uni(input):
    '''
      Parameters:
        input : number of neuron connections coming into the layer
      Returns:
        A float64 which represents the he uniform factor
      Function:
        finds the he uniform factor with the data type as float64 with the given number of input connections
    '''
    return np.sqrt(6 / input).astype(np.float64)


def clip_value(input, range):
    '''
      Parameters:
        input : the value on which to perform the clipping
        range : the extreme values on which the clipping has to be performed
      Returns:
        The same array after performing the clipping
      Function:
        performs clipping (if the input value is beyond the range then increase or decrease to the extreme value) of the input
    '''
    return np.clip(input, -range, range)


def exponent(input):
    '''
      Parameters:
        input : the value for which to find the exponent
      Returns:
        The exponent value
      Function:
        finds the exponent of the input
    '''
    return np.exp(input)


def create_ones(dim1, dim2):
    '''
      Parameters:
        dim1 : dimension value (number of rows) for the array
        dim2 : dimension value (number of columns) for the array
      Returns:
        A 2D numpy array
      Function:
        creates a 2D array of all ones
    '''
    return np.ones((dim1, dim2))


def create_zeros2D(dim1, dim2):
    '''
      Parameters:
        dim1 : dimension value (number of rows) for the array
        dim2 : dimension value (number of columns) for the array
      Returns:
        A 2D numpy array
      Function:
        creates a 2D array of all zeros with the datatype as float64
    '''
    return np.zeros((dim1, dim2), dtype = np.float64)


def create_zeros1D(dim):
    '''
      Parameters:
        dim : dimension value (number of columns) for the array
      Returns:
        A 1D numpy array
      Function:
        creates a 2D array of all zeros with the datatype as float64
    '''
    return np.zeros(dim, dtype = np.float64)


def make_row_vector(input):
    '''
      Parameters:
        input : the input array which has to be reshaped
      Returns:
        A 1D numpy array
      Function:
        creates a 1D array from the input by reducing its dimension
    '''
    return input.reshape(1, -1)


def expand_by_repeating(input, dim):
    '''
      Parameters:
        input : the input array which has to be modified
        dim : the target dimension
      Returns:
        A 2D numpy array
      Function:
        creates a 2D array by repeating the columns of the input array
    '''
    return np.repeat(input, dim, axis = 0)


def multiply(matrix1, matrix2):
    '''
      Parameters:
        matrix1 : the first array
        matrix2 : the second array
      Returns:
        A numpy array
      Function:
        multiplies (matrix multiplication) two matrices
    '''
    return np.matmul(matrix1, matrix2)


def matrixMultiply(matrix1, matrix2):
    '''
      Parameters:
        matrix1 : the first array
        matrix2 : the second array
      Returns:
        A numpy array
      Function:
        multiplies (hadamard prouduct) two matrices
    '''
    return np.multiply(matrix1, matrix2)


def checkIfNone(passedObject):
    '''
      Parameters:
        passedObject : the object which has to be checked for null
      Returns:
        A boolean representing if the object is null or not
      Function:
        checks if the object is null or not (returns false if it is null)
    '''
    isNone = not(passedObject is None)
    return isNone