import numpy as np

'''
    The given images are 2D (28 x 28)
    But the input to the neural network has to be a 1D array
    So this function flattens the 2D images into a 1D array
    Also this function normalizes each pixel of the image (because the input of each pixel is in the range 0-256)
    This noramlization is needed to handle avoid exploding of the gradients
'''
def flattenAndNormalizeInputImage(images):
    '''
      Parameters:
        images : input images (represented as 28 x 28 matrices)
      Returns:
        normalized_images : a vector of size 256 representing the normalized and flattened form of the image
      Function:
        flattenes and normalies the input images
    '''

    size_of_image_vector = images.shape[0]

    # flatten the image
    flattened_images = images.reshape(size_of_image_vector, -1)

    # normalize the image
    normalized_images = flattened_images / 255.0

    return normalized_images


'''
    The given output is just an integer denoting the index of the output class
    To apply the softmax function on the output class, we need to convert the output into a one hot vector
    This function onverts the given 1D output vector into a 2D vector with each row representing a one hot vector of that output class
'''
def oneHotEncodeOutput(output):
    '''
      Parameters:
        output : output class label (an integer representing the output class index)
      Returns:
        one_hot_2D_np_array : a matrix of vectors of size 10 each which is the one hot form of the input
      Function:
        creates one hot vectors for the outputs
    '''

    output_one_hot_2d_vector = list()

    for output_class in output:
        # create a vector of all zeros size 10
        one_hot_vector = np.zeros(10)

        # mark the index as 1 which is present in the output label
        one_hot_vector[output_class] = 1

        output_one_hot_2d_vector.append(one_hot_vector)
    
    one_hot_2D_np_array = np.array(output_one_hot_2d_vector)
      
    return one_hot_2D_np_array