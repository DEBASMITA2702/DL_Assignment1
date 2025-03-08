import numpy as np
from ForwardPropagation import forwardPropagationForPredictions
from Accuracy import trainingAccuracy
from LossFunctions import crossEntropyLoss, meanSquaredLoss

def calculateTrainingAccuracyandLoss(x_train, y_train, weights, bias, num_total_layers, activation_function, loss_function):
    predictions = list()
    data_size = x_train.shape[0]
    for point in range(data_size):
        _, _, predictedValues = forwardPropagationForPredictions(weights, bias, x_train[point], num_total_layers, activation_function)
        predictions.append(predictedValues)
    training_accuracy = trainingAccuracy(y_train, np.array(predictions))
    
    training_loss = 0.0
    if(loss_function == "cross_entropy"):
        training_loss = crossEntropyLoss(y_train, np.array(predictions)) / x_train.shape[0]
    else:
        training_loss = meanSquaredLoss(y_train,np.array(predictions))
    
    return training_accuracy, training_loss

def calculateValidationAccuracyandLoss(x_validation, y_validation, weights, bias, num_total_layers, activation_function, loss_function):
    predictions=list()
    data_size = x_validation.shape[0]
    for i in range(x_validation.shape[0]):
        _, _, predictedValues = forwardPropagationForPredictions(weights, bias, x_validation[i], num_total_layers, activation_function)
        predictions.append(predictedValues)
    validation_accuracy = trainingAccuracy(y_validation, np.array(predictions))

    validation_loss = 0.0
    if(loss_function == "cross_entropy"):
        validation_loss = crossEntropyLoss(y_validation, np.array(predictions)) / y_validation.shape[0]
    else:
        validation_loss = meanSquaredLoss(y_validation, np.array(predictions))
    
    return validation_accuracy, validation_loss

