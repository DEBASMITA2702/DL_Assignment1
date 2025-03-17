import wandb
import numpy as np
from keras.datasets import mnist,fashion_mnist
from ParseArguments import collectArguments
from SetHyperparamValues import setValuesFromArgument
from LoadDataset import loadAndSplitDataset
from FeedForwardNeuralNetwork import NeuralNetwork
from ForwardPropagation import forwardPropagationForPredictions
from DataPreProcessing import flattenAndNormalizeInputImage, oneHotEncodeOutput
from Accuracy import trainingAccuracy
from ConfusionMatrix import createAndPlotConfusionMatrix

if __name__ == '__main__':
    
    # collect the arguments from command line
    argument_passed = collectArguments()

    # create the hyperparameter dictionary
    hyperparam_dictionary = setValuesFromArgument(argument_passed)

    # loads the dataset and splits it to training, validation and test sets based on the choice of the dataset
    x_train, y_train, x_val, y_val, x_test, y_test = loadAndSplitDataset(fashion_mnist)
    if (hyperparam_dictionary.get("dataset_name") == "mnist"):
        x_train, y_train, x_val, y_val, x_test, y_test = loadAndSplitDataset(mnist)

    # login to wandb
    wandb.login()

    # initialoze the project
    wandb.init(project = hyperparam_dictionary.get("project_name"), entity = hyperparam_dictionary.get("entity_name"))

    # create the neural network object based on the values from the hyperparameter dictionary
    TrainPy_Network_model = NeuralNetwork(x_train, y_train, x_val, y_val, epochs = hyperparam_dictionary.get("epochs"), 
                                  num_hidden_layers = hyperparam_dictionary.get("hidden_layers"), 
                                  neurons_in_hidden_layer = hyperparam_dictionary.get("neurons_in_hidden"), 
                                  initialization_method = hyperparam_dictionary.get("weight_initialization"), 
                                  activation_function = hyperparam_dictionary.get("activation"), 
                                  loss_function = hyperparam_dictionary.get("loss_type"), 
                                  learning_rate = hyperparam_dictionary.get("learning_rate"), 
                                  weight_decay = hyperparam_dictionary.get("weight_decay")
                                )
    
    # call the fitting function
    learned_weights, learned_biases = TrainPy_Network_model.fitModel(beta = hyperparam_dictionary.get("beta"), 
                                                             beta1 = hyperparam_dictionary.get("beta1"), 
                                                             beta2 = hyperparam_dictionary.get("beta2"),
                                                             epsilon = hyperparam_dictionary.get("epsilon"), 
                                                             optimizer = hyperparam_dictionary.get("optimizer"), 
                                                             batch_size = hyperparam_dictionary.get("batch_size"),
                                                             trainPy = True,
                                                             momentum = hyperparam_dictionary.get("momentum")
                                                            )
    
    # run the model on the test data and gather the predictions of the model
    x_test_flattened = flattenAndNormalizeInputImage(x_test)
    y_test_one_hot = oneHotEncodeOutput(y_test)
    
    predictions = list()
    data_size = y_test_one_hot.shape[0]
    for point in range(data_size):
        _, _, predictedValues = forwardPropagationForPredictions(learned_weights, learned_biases, x_test_flattened[point], 
                                                                TrainPy_Network_model.num_total_layers, 
                                                                TrainPy_Network_model.activation_function)
        predictions.append(predictedValues)
    

    # if test == 1
    if hyperparam_dictionary.get("test") == 1:
        test_accuracy = trainingAccuracy(y_test_one_hot, np.array(predictions))
        
        '''print the test accuracy'''
        print("\nTest accuracy is : ", test_accuracy)
    
    # if confusion matrix == 1
    if hyperparam_dictionary.get("confusion_matrix"):
        createAndPlotConfusionMatrix(y_test_one_hot, predictions, dataset = hyperparam_dictionary.get("dataset_name"))
    
    # finish wandb
    wandb.finish()