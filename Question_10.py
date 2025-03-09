import numpy as np
from LoadDataset import loadAndSplitDataset
from FeedForwardNeuralNetwork import NeuralNetwork
from DataPreProcessing import flattenAndNormalizeInputImage, oneHotEncodeOutput
from ForwardPropagation import forwardPropagationForPredictions
from Accuracy import trainingAccuracy
from keras.datasets import mnist

if __name__ == '__main__':
    '''loads the dataset'''
    x_train, y_train, x_val, y_val, x_test, y_test = loadAndSplitDataset(mnist)

    '''flatten input and one hot encode output'''
    x_test_flattened = flattenAndNormalizeInputImage(x_test)
    y_test_one_hot = oneHotEncodeOutput(y_test)


    '''
        1st model
        This is the best model obtained from sweeping
        The configuration is :
            activation function : relu
            batch size : 128
            beta value : 0.999
            initialization method : he normal
            learning rate : 0.005
            neurons in each hidden layer : 128
            number of epochs : 10
            number of hidden layers : 3
            optimization alogrithm : adam
            loss function = cross_entropy
            weight decay : 0.0005
    '''
    print("\n\n1st Model")
    First_model = NeuralNetwork(x_train, y_train, x_val, y_val, epochs = 10, num_hidden_layers = 3, neurons_in_hidden_layer = 128, 
                                  initialization_method = "he_nor", activation_function = "relu", loss_function = "cross_entropy", 
                                  learning_rate = 0.005, weight_decay = 0.0005)

    learned_weights, learned_biases = First_model.fitModel(beta = 0.999, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-5, optimizer = "adam", 
                           batch_size = 128, mnistTestRun = True)
    
    '''get the predictions of the test images based on the weights and biases learned'''
    predictions = list()
    data_size = y_test_one_hot.shape[0]
    for point in range(data_size):
        _, _, predictedValues = forwardPropagationForPredictions(learned_weights, learned_biases, x_test_flattened[point], 
                                                                 First_model.num_total_layers, 
                                                                 First_model.activation_function)
        predictions.append(predictedValues)
    
    test_accuracy = trainingAccuracy(y_test_one_hot, np.array(predictions))

    '''print the test accuracy'''
    print("Test Accuracy for Mnist dataset is : ", test_accuracy)



    '''
    2nd model
    This is the second best model obtained from sweeping
    The configuration is :
        activation function : relu
        batch size : 256
        beta value : 0.999
        initialization method : he uniform
        learning rate : 0.005
        neurons in each hidden layer : 128
        number of epochs : 20
        number of hidden layers : 3
        optimization alogrithm : nadam
        loss function = cross_entropy
        weight decay : 0.0005
    '''
    print("\n\n2nd Model")
    First_model = NeuralNetwork(x_train, y_train, x_val, y_val, epochs = 20, num_hidden_layers = 3, neurons_in_hidden_layer = 128, 
                                initialization_method = "he_uni", activation_function = "relu", loss_function = "cross_entropy", 
                                learning_rate = 0.005, weight_decay = 0.0005)

    learned_weights, learned_biases = First_model.fitModel(beta = 0.999, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-5, optimizer = "nadam", 
                        batch_size = 256, mnistTestRun = True)
    
    '''get the predictions of the test images based on the weights and biases learned'''
    predictions=list()
    data_size = y_test_one_hot.shape[0]
    for point in range(data_size):
        _, _, predictedValues = forwardPropagationForPredictions(learned_weights, learned_biases, x_test_flattened[point], 
                                                                First_model.num_total_layers, 
                                                                First_model.activation_function)
        predictions.append(predictedValues)
    
    test_accuracy = trainingAccuracy(y_test_one_hot, np.array(predictions))

    '''print the test accuracy'''
    print("Test Accuracy for Mnist dataset is : ", test_accuracy)



    '''
    3rd model
    This is the third best model obtained from sweeping
    The configuration is :
        activation function : relu
        batch size : 256
        beta value : 0.999
        initialization method : he normal
        learning rate : 0.005
        neurons in each hidden layer : 32
        number of epochs : 20
        number of hidden layers : 5
        optimization alogrithm : nadam
        loss function = cross_entropy
        weight decay : 0
    '''
    print("\n\n3rd Model")
    First_model = NeuralNetwork(x_train, y_train, x_val, y_val, epochs = 20, num_hidden_layers = 5, neurons_in_hidden_layer = 32, 
                                initialization_method = "he_nor", activation_function = "relu", loss_function = "cross_entropy", 
                                learning_rate = 0.005, weight_decay = 0)

    learned_weights, learned_biases = First_model.fitModel(beta = 0.999, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-5, optimizer = "nadam", 
                        batch_size = 256, mnistTestRun = True)
    
    '''get the predictions of the test images based on the weights and biases learned'''
    predictions=list()
    data_size = y_test_one_hot.shape[0]
    for point in range(data_size):
        _, _, predictedValues = forwardPropagationForPredictions(learned_weights, learned_biases, x_test_flattened[point], 
                                                                First_model.num_total_layers, 
                                                                First_model.activation_function)
        predictions.append(predictedValues)
    
    test_accuracy = trainingAccuracy(y_test_one_hot, np.array(predictions))

    '''print the test accuracy'''
    print("Test Accuracy for Mnist dataset is : ", test_accuracy)