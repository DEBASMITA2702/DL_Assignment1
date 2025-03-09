import wandb
import numpy as np
from LoadDataset import loadAndSplitDataset
from FeedForwardNeuralNetwork import NeuralNetwork
from DataPreProcessing import flattenAndNormalizeInputImage, oneHotEncodeOutput
from ForwardPropagation import forwardPropagationForPredictions
from Accuracy import trainingAccuracy
from ConfusionMatrix import createAndPlotConfusionMatrix
from keras.datasets import fashion_mnist

if __name__ == '__main__':
    '''loads the dataset'''
    x_train, y_train, x_val, y_val, x_test, y_test = loadAndSplitDataset(fashion_mnist)

    '''flatten input and one hot encode output'''
    x_test_flattened = flattenAndNormalizeInputImage(x_test)
    y_test_one_hot = oneHotEncodeOutput(y_test)

    '''
        create a neural network object with the best configuration
        the best configuration is:
            activation function : relu
            batch size : 128
            beta value : 0.999
            initialization method : he normal
            learning rate : 0.005
            neurons in each hidden layer : 128
            number of epochs : 10
            number of hidden layers : 3
            optimization alogrithm : adam
            weight decay : 0.0005
    '''
    Best_network_model = NeuralNetwork(x_train, y_train, x_val, y_val, epochs = 10, num_hidden_layers = 3, neurons_in_hidden_layer = 128, 
                                  initialization_method = "he_nor", activation_function = "relu", loss_function = "cross_entropy", 
                                  learning_rate = 0.005, weight_decay = 0.0005)

    learned_weights, learned_biases = Best_network_model.fitModel(beta = 0.999, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-5, optimizer = "adam", 
                           batch_size = 128, bestConfigTestAccuracyRun = True)
    

    '''get the predictions of the test images based on the weights and biases learned'''
    predictions = list()
    data_size = y_test_one_hot.shape[0]
    for point in range(data_size):
        _, _, predictedValues = forwardPropagationForPredictions(learned_weights, learned_biases, x_test_flattened[point], 
                                                                 Best_network_model.num_total_layers, 
                                                                 Best_network_model.activation_function)
        predictions.append(predictedValues)
    
    test_accuracy = trainingAccuracy(y_test_one_hot, np.array(predictions))


    '''print the test accuracy'''
    print("Test accuracy for the best configuration model is : ", test_accuracy)



    # '''call confusion matrix to create and plot the confusion matrix'''
    wandb.login()
    wandb.init(project = "Debasmita-DA6410-Assignment-1")
    createAndPlotConfusionMatrix(y_test_one_hot, predictions, dataset = "fashion_mnist")
    wandb.finish()