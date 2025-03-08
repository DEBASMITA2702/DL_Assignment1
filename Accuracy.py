import numpy as np
from ForwardPropagation import forwardPropagation

def trainingAccuracy(true_output, predicted_output):
    accurate_predictions = 0
    data_size = predicted_output.shape[0]

    for point in range(data_size):
        true_class = np.argmax(true_output[point])
        predicted_class = np.argmax(predicted_output[point])
                               
        if true_class == predicted_class:
            accurate_predictions += 1

    accuracy_ratio = accurate_predictions / data_size
    
    return accuracy_ratio


def validationAccuracy(weights, biases, input, num_total_layers, activation_function, true_output):
    accurate_predictions = 0
    
    '''doing forward propagation to find out the predictions of the input according to the weight and bias'''
    _, _, predicted_output = forwardPropagation(weights, biases, input, num_total_layers, activation_function)

    data_size = true_output.shape[0]

    for point in range(data_size):
        '''checking if the true class value and the predicted class are same or not'''
        true_class = np.argmax(true_output[point])
        predicted_class = np.argmax(predicted_output[point])

        if true_class == predicted_class:
            accurate_predictions+=1

    accuracy_ratio = accurate_predictions / data_size

    return accuracy_ratio