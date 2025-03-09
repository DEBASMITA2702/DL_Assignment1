import numpy as np

'''return the shape of the 1st dimension of an numpy array'''
def getShape0(npArray):
    return npArray.shape[0]

'''generates and returns a numpy array of random numbers in a range'''
def generateRandom(range1, range2, replaceParam):
    return np.random.choice(range1, range2, replace = replaceParam)

'''deletes some given data from another data in a numpy array'''
def deleteData(sourceData, deleteData):
    return np.delete(sourceData, deleteData, axis = 0)

'''creates a random 1D vector of given dimension formed'''
def random1DFloat(dim1):
    return np.random.randn(dim1).astype(np.float64)

'''creates a random 2D vector of given dimension formed'''
def random2D(dim1, dim2):
    return np.random.randn(dim1, dim2)

'''creates a random 2D vector of given dimension formed of floats'''
def random2DFloat(dim1, dim2):
    return np.random.randn(dim1, dim2).astype(np.float64)

'''returns the factor for xavier normal initialization'''
def xav_nor(input, output):
    return np.sqrt(2 / (input + output)).astype(np.float64)

'''returns the factor for xavier uniform initialization'''
def xav_uni(input, output):
    return np.sqrt(6 / (input + output)).astype(np.float64)

'''returns the factor for he normal initialization'''
def he_nor(input):
    return np.sqrt(2 / input).astype(np.float64)

'''returns the factor for he uniform initialization'''
def he_uni(input):
    return np.sqrt(6 / input).astype(np.float64)

'''clips the value within a range'''
def clip_value(input, range):
    return np.clip(input, -range, range)

'''finds exponent of input'''
def exponent(input):
    return np.exp(input)

'''creates an array of all ones'''
def create_ones(dim1, dim2):
    return np.ones((dim1, dim2))

'''creates an 2D array of all ones'''
def create_zeros2D(dim1, dim2):
    return np.zeros((dim1, dim2), dtype=np.float64)

'''creates an 1D array of all ones'''
def create_zeros1D(dim):
    return np.zeros(dim, dtype=np.float64)

'''makes a row vector'''
def make_row_vector(input):
    return input.reshape(1, -1)

'''expands a vector into a matrix by repeating the rows'''
def expand_by_repeating(input, dim):
    return np.repeat(input, dim, axis = 0)

'''multiplies two matrices'''
def multiply(matrix1, matrix2):
    return np.matmul(matrix1, matrix2)

'''multiply two matrices'''
def matrixMultiply(matrix1, matrix2):
    return np.multiply(matrix1, matrix2)

'''checks if the passed obejct is None or not'''
def checkIfNone(passedObject):
    isNone = not(passedObject is None)
    return isNone