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