import numpy as np
from ForwardPropagation import forwardPropagationForPredictions
from Accuracy import trainingAccuracy
from LossFunctions import crossEntropyLoss, meanSquaredLoss

''' these function are intended to find the loss and accuracy for the different data provided (training, validation and test)'''

def calculateTrainingAccuracyandLoss(x_train, y_train, weights, bias, num_total_layers, activation_function, loss_function):
    '''
      Parameters:
        x_train : training input data
        y_train : true output provided with the datase for the corresponding input data
        weights : weights learned
        bias : biases learned
        num_total_layers : total number of layers in the framework
        activation_function : choice of activation function used in the framework
        loss_function : choice of loss function used in the framework
      Returns:
        training_accuracy : a float representing the accuracy calculated for the data
        training_loss : a float representing the loss calculated for the data
      Function:
        finds and returns the accuracy and loss calculated for the data
    '''

    # for finding accuracy we will run the forward propagation algorithm once to find the predicted values of the input with the current learned weights and biases
    predictions = list()
    data_size = x_train.shape[0]

    # going over the dataset one data point at a time
    for point in range(data_size):
        # running forward propagation and storing the predictions
        _, _, predictedValues = forwardPropagationForPredictions(weights, bias, x_train[point], num_total_layers, activation_function)
        predictions.append(predictedValues)
    
    # finding the accuracy
    training_accuracy = trainingAccuracy(y_train, np.array(predictions))
    
    # for finding the loss we use the predictions that are found above
    training_loss = 0.0
    # call the corresponding function based on the choice of loss function
    if(loss_function == "cross_entropy"):
        training_loss = crossEntropyLoss(y_train, np.array(predictions)) / x_train.shape[0]
    else:
        training_loss = meanSquaredLoss(y_train,np.array(predictions))
    
    # return the values
    return training_accuracy, training_loss


def calculateValidationAccuracyandLoss(x_validation, y_validation, weights, bias, num_total_layers, activation_function, loss_function):
    '''
      Parameters:
        x_validation : validation input data
        y_validation : true output provided with the datase for the corresponding input data
        weights : weights learned
        bias : biases learned
        num_total_layers : total number of layers in the framework
        activation_function : choice of activation function used in the framework
        loss_function : choice of loss function used in the framework
      Returns:
        validation_accuracy : a float representing the accuracy calculated for the data
        validation_loss : a float representing the loss calculated for the data
      Function:
        finds and returns the accuracy and loss calculated for the data
    '''

    # for finding accuracy we will run the forward propagation algorithm once to find the predicted values of the input with the current learned weights and biases
    predictions = list()
    data_size = x_validation.shape[0]

    # going over the dataset one data point at a time
    for point in range(data_size):
        _, _, predictedValues = forwardPropagationForPredictions(weights, bias, x_validation[point], num_total_layers, activation_function)
        predictions.append(predictedValues)
    
    # finding the accuracy
    validation_accuracy = trainingAccuracy(y_validation, np.array(predictions))

    # for finding the loss we use the predictions that are found above
    validation_loss = 0.0
    # call the corresponding function based on the choice of loss function
    if(loss_function == "cross_entropy"):
        validation_loss = crossEntropyLoss(y_validation, np.array(predictions)) / y_validation.shape[0]
    else:
        validation_loss = meanSquaredLoss(y_validation, np.array(predictions))
    
    # return the values
    return validation_accuracy, validation_loss

def calculateTestAccuracyandLoss(x_test, y_test, weights, bias, num_total_layers, activation_function, loss_function):
    '''
      Parameters:
        x_test : test input data
        y_test : true output provided with the datase for the corresponding input data
        weights : weights learned
        bias : biases learned
        num_total_layers : total number of layers in the framework
        activation_function : choice of activation function used in the framework
        loss_function : choice of loss function used in the framework
      Returns:
        test_accuracy : a float representing the accuracy calculated for the data
        test_loss : a float representing the loss calculated for the data
      Function:
        finds and returns the accuracy and loss calculated for the data
    '''

    # for finding accuracy we will run the forward propagation algorithm once to find the predicted values of the input with the current learned weights and biases
    predictions = list()
    data_size = x_test.shape[0]

    # going over the dataset one data point at a time
    for point in range(data_size):
        _, _, predictedValues = forwardPropagationForPredictions(weights, bias, x_test[point], num_total_layers, activation_function)
        predictions.append(predictedValues)

    # finding the accuracy
    test_accuracy = trainingAccuracy(y_test, np.array(predictions))

    # for finding the loss we use the predictions that are found above
    test_loss = 0.0
    # call the corresponding function based on the choice of loss function
    if(loss_function == "cross_entropy"):
        test_loss = crossEntropyLoss(y_test, np.array(predictions)) / y_test.shape[0]
    else:
        test_loss = meanSquaredLoss(y_test, np.array(predictions))
    
    # return the values
    return test_accuracy, test_loss