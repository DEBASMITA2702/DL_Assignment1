import numpy as np


'''Even though this function is named as trainingAccuracy, it calculates the accuracy of any type of data (training, validation or test)'''
def trainingAccuracy(true_output, predicted_output):
    '''
      Parameters:
        true_output : a 2D dimensional (of data size x number of output classes) np array containing the true one hot test vectors
        predicted_output : a 2D dimensional (of data size x number of output classes) np array containing the result of the softmax functions
      Returns:
        accuracy precentage of correct predictionsin the data
      Function:
        Performs accuracy calculation of data
    '''
    # variable to calculate the number of correct predictions in the batch
    accurate_predictions = 0

    # variable to store the number of data points
    data_size = predicted_output.shape[0]

    for point in range(data_size):
        # find the true class index (0 to 9) for the current point
        true_class = np.argmax(true_output[point])
        # find the predicted class index (0 to 9) for the current point (the maxmimum one from the softmax function is the predicted class)
        predicted_class = np.argmax(predicted_output[point])
        
        # if both true and predicted class indices are same then it is considered as a correct prediction
        if true_class == predicted_class:
            accurate_predictions += 1

    # find the accuracy ratio
    accuracy_ratio = accurate_predictions / data_size
    
    return accuracy_ratio