import wandb
import DataPreProcessing
import WeightsAndBiasesAndGradientsInitialization
from CalculateAccuracyAndLoss import calculateTrainingAccuracyandLoss, calculateValidationAccuracyandLoss
from WeightsAndBiasesAndGradientsInitialization import initializeGradients
import OptimizationFunctions
import UpdateWeightsAndBiases
from WandbLog import printAndLogAccuracyAndLoss, printAccuracyForMnistTest

''' 
    this is the class representing the neural network model
    the whole program is driven from this class
'''

class NeuralNetwork:
    def __init__(self, x_train, y_train, x_validation, y_validation, epochs, num_hidden_layers, neurons_in_hidden_layer, initialization_method, activation_function, loss_function, learning_rate, weight_decay):
        
        '''
        Parameters:
            self : the self object of the class
            x_train : training input data
            y_train : training output labels
            x_validation : validation input data
            y_validation : validation output labels
            epochs : number of epochs for the model
            num_hidden_layers : number of hidden layers in the model
            neurons_in_hidden_layer : number of neurons in each hidden layer in the model
            initialization_method : initialization method for the model
            activation_function : activation function for the model
            loss_function : loss function for the model
            learning_rate : learning rate of the model
            weight_decay : weight decay of the model
        Returns:
            none
        Function:
            this is the constructor that initalizes the different variables of the model object
        '''

        # set the hyperparameters of the model
        self.epochs = epochs
        self.num_hidden_layers = num_hidden_layers
        self.neurons_in_hidden_layers = neurons_in_hidden_layer
        self.initialization_method = initialization_method
        self.activation_function = activation_function
        self.loss_function = loss_function
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_total_layers = self.num_hidden_layers + 2


        # prepare the data (flattening for inputs and one hot encoding for outputs)
        self.x_train = DataPreProcessing.flattenAndNormalizeInputImage(x_train)
        self.y_train = DataPreProcessing.oneHotEncodeOutput(y_train)
        self.x_validation = DataPreProcessing.flattenAndNormalizeInputImage(x_validation)
        self.y_validation = DataPreProcessing.oneHotEncodeOutput(y_validation)


        # initialize the weight and bias parameters of the model
        input_length_for_weight_and_bias_initialization = self.x_train.shape[1]
        output_length_for_weight_and_bias_initialization = self.y_train.shape[1]
        self.weights = WeightsAndBiasesAndGradientsInitialization.initializeWeight(self.initialization_method, self.num_total_layers, 
                                                                       self.neurons_in_hidden_layers, 
                                                                       input_length_for_weight_and_bias_initialization, 
                                                                       output_length_for_weight_and_bias_initialization)
        self.bias = WeightsAndBiasesAndGradientsInitialization.initializeBias(self.initialization_method, self.num_total_layers, 
                                                                  self.neurons_in_hidden_layers,
                                                                  output_length_for_weight_and_bias_initialization)



    def runStochastic(self, batch_size, bestConfigTestAccuracyRun, mnistTestRun):
        '''
        Parameters:
            self : the self object of the class
            batch_size : batch size used for the model
            bestConfigTestAccuracyRun : a boolean representing whether this model is used for finding test accuracy of the best model (Question 7)
            mnistTestRun : a boolean representing whether this model is used for finding the performance on Mnist dataset (Question 10)
        Returns:
            weights : weights learned
            bias : biases learned
        Function:
            runs the stochastic gradient descent algorithm
        '''

        # run over the epochs
        for epoch in range(self.epochs):
            # run the algorithm to find the weights and biases in the epoch
            self.weights, self.bias = OptimizationFunctions.stochastic_gradient_descent(self, batch_size)

            if bestConfigTestAccuracyRun == False:
                # find accuracy and loss
                trainingAccuracyCurrentEpoch, trainingLossCurrentEpoch = calculateTrainingAccuracyandLoss(self.x_train, self.y_train, self.weights, self.bias, self.num_total_layers, self.activation_function, self.loss_function)
                validationAccuracyCurrentEpoch, validationLossCurrentEpoch = calculateValidationAccuracyandLoss(self.x_validation, self.y_validation, self.weights, self.bias, self.num_total_layers, self.activation_function, self.loss_function)

                if mnistTestRun == False:
                    # log the accuracy and loss to wandb
                    printAndLogAccuracyAndLoss(epoch, trainingAccuracyCurrentEpoch, trainingLossCurrentEpoch, validationAccuracyCurrentEpoch, validationLossCurrentEpoch)

                if mnistTestRun == True and epoch == self.epochs - 1:
                    # print the accuracy for mnist dataset (question 10)
                    printAccuracyForMnistTest(trainingAccuracyCurrentEpoch, validationAccuracyCurrentEpoch)
        
        return self.weights, self.bias


    def runMomentum(self, beta, batch_size, bestConfigTestAccuracyRun, mnistTestRun):
        '''
        Parameters:
            self : the self object of the class
            beta : momentum rate for the model
            batch_size : batch size used for the model
            bestConfigTestAccuracyRun : a boolean representing whether this model is used for finding test accuracy of the best model (Question 7)
            mnistTestRun : a boolean representing whether this model is used for finding the performance on Mnist dataset (Question 10)
        Returns:
            weights : weights learned
            bias : biases learned
        Function:
            runs the momentum based gradient descent algorithm
        '''

        # initialize the history
        history_weights, history_biases = initializeGradients(self.num_total_layers, self.neurons_in_hidden_layers, self.x_train.shape[1], self.y_train.shape[1])

        # run over the epochs
        for epoch in range(self.epochs):
            # run the algorithm to find the weights and biases in the epoch
            self.weights, self.bias, history_weights, history_biases = OptimizationFunctions.momentum_based_gradient_descent(self, history_weights, history_biases, beta, batch_size)

            if bestConfigTestAccuracyRun == False:
                # find accuracy and loss
                trainingAccuracyCurrentEpoch, trainingLossCurrentEpoch = calculateTrainingAccuracyandLoss(self.x_train, self.y_train, self.weights, self.bias, self.num_total_layers, self.activation_function, self.loss_function)
                validationAccuracyCurrentEpoch, validationLossCurrentEpoch = calculateValidationAccuracyandLoss(self.x_validation, self.y_validation, self.weights, self.bias, self.num_total_layers, self.activation_function, self.loss_function)
                
                if mnistTestRun == False:
                    # log the accuracy and loss to wandb
                    printAndLogAccuracyAndLoss(epoch, trainingAccuracyCurrentEpoch, trainingLossCurrentEpoch, validationAccuracyCurrentEpoch, validationLossCurrentEpoch)

                if mnistTestRun == True and epoch == self.epochs - 1:
                    # print the accuracy for mnist dataset (question 10)
                    printAccuracyForMnistTest(trainingAccuracyCurrentEpoch, validationAccuracyCurrentEpoch)
        
        return self.weights, self.bias
    

    def runNesterov(self, beta, batch_size, bestConfigTestAccuracyRun, mnistTestRun):
        '''
        Parameters:
            self : the self object of the class
            beta : momentum rate for the model
            batch_size : batch size used for the model
            bestConfigTestAccuracyRun : a boolean representing whether this model is used for finding test accuracy of the best model (Question 7)
            mnistTestRun : a boolean representing whether this model is used for finding the performance on Mnist dataset (Question 10)
        Returns:
            weights : weights learned
            bias : biases learned
        Function:
            runs the nesterov accelerated gradient descent algorithm
        '''

        # initialize the history
        history_weights, history_biases = initializeGradients(self.num_total_layers, self.neurons_in_hidden_layers, self.x_train.shape[1], self.y_train.shape[1])

        # run over the epochs
        for epoch in range(self.epochs):
            # initialization for the nesterov algorithm
            uw, ub=dict(),dict()
            for i in range(1, len(history_weights)):
                uw[i] = beta * history_weights[i]
                ub[i] = beta * history_biases[i]

            inital_weights, inital_biases = UpdateWeightsAndBiases.momentum_update(self.weights, self.bias, uw, ub, self.learning_rate, self.weight_decay)

            # run the algorithm to find the weights and biases in the epoch
            self.weights, self.bias, history_weights, history_biases = OptimizationFunctions.nesterov_gradient_descent(self, uw, ub, history_weights, history_biases, inital_weights, inital_biases, beta, batch_size)

            if bestConfigTestAccuracyRun == False:
                # find accuracy and loss
                trainingAccuracyCurrentEpoch, trainingLossCurrentEpoch = calculateTrainingAccuracyandLoss(self.x_train, self.y_train, self.weights, self.bias, self.num_total_layers, self.activation_function, self.loss_function)
                validationAccuracyCurrentEpoch, validationLossCurrentEpoch = calculateValidationAccuracyandLoss(self.x_validation, self.y_validation, self.weights, self.bias, self.num_total_layers, self.activation_function, self.loss_function)

                if mnistTestRun == False:
                    # log the accuracy and loss to wandb
                    printAndLogAccuracyAndLoss(epoch, trainingAccuracyCurrentEpoch, trainingLossCurrentEpoch, validationAccuracyCurrentEpoch, validationLossCurrentEpoch)

                if mnistTestRun == True and epoch == self.epochs - 1:
                    # print the accuracy for mnist dataset (question 10)
                    printAccuracyForMnistTest(trainingAccuracyCurrentEpoch, validationAccuracyCurrentEpoch)
        
        return self.weights, self.bias
    

    def runRmsprop(self, beta, epsilon, batch_size, bestConfigTestAccuracyRun, mnistTestRun):
        '''
        Parameters:
            self : the self object of the class
            beta : beta value for the model
            epsilon : epsilon value for the model
            batch_size : batch size used for the model
            bestConfigTestAccuracyRun : a boolean representing whether this model is used for finding test accuracy of the best model (Question 7)
            mnistTestRun : a boolean representing whether this model is used for finding the performance on Mnist dataset (Question 10)
        Returns:
            weights : weights learned
            bias : biases learned
        Function:
            runs the rmsprop gradient descent algorithm
        '''

        # initialize the history
        history_weights, history_biases = initializeGradients(self.num_total_layers, self.neurons_in_hidden_layers, self.x_train.shape[1], self.y_train.shape[1])

        # run over the epochs
        for epoch in range(self.epochs):
            # run the algorithm to find the weights and biases in the epoch
            self.weights, self.bias, history_weights, history_biases = OptimizationFunctions.rmsprop_gradient_descent(self, history_weights, history_biases, beta, epsilon, batch_size)
            
            if bestConfigTestAccuracyRun == False:
                # find accuracy and loss
                trainingAccuracyCurrentEpoch, trainingLossCurrentEpoch = calculateTrainingAccuracyandLoss(self.x_train, self.y_train, self.weights, self.bias, self.num_total_layers, self.activation_function, self.loss_function)
                validationAccuracyCurrentEpoch, validationLossCurrentEpoch = calculateValidationAccuracyandLoss(self.x_validation, self.y_validation, self.weights, self.bias, self.num_total_layers, self.activation_function, self.loss_function)

                if mnistTestRun == False:
                    # log the accuracy and loss to wandb
                    printAndLogAccuracyAndLoss(epoch, trainingAccuracyCurrentEpoch, trainingLossCurrentEpoch, validationAccuracyCurrentEpoch, validationLossCurrentEpoch)
                
                if mnistTestRun == True and epoch == self.epochs - 1:
                    # print the accuracy for mnist dataset (question 10)
                    printAccuracyForMnistTest(trainingAccuracyCurrentEpoch, validationAccuracyCurrentEpoch)
        
        return self.weights, self.bias
    

    def runAdam(self, beta1, beta2, epsilon, batch_size, bestConfigTestAccuracyRun, mnistTestRun):
        '''
        Parameters:
            self : the self object of the class
            beta1 : beta1 value for the model
            beta2 : beta2 value for the model
            epsilon : epsilon value for the model
            batch_size : batch size used for the model
            bestConfigTestAccuracyRun : a boolean representing whether this model is used for finding test accuracy of the best model (Question 7)
            mnistTestRun : a boolean representing whether this model is used for finding the performance on Mnist dataset (Question 10)
        Returns:
            weights : weights learned
            bias : biases learned
        Function:
            runs the adam gradient descent algorithm
        '''

        # initialize the parameters for the adam algorithm
        vw, vb = initializeGradients(self.num_total_layers, self.neurons_in_hidden_layers, self.x_train.shape[1], self.y_train.shape[1])
        mw, mb = initializeGradients(self.num_total_layers, self.neurons_in_hidden_layers, self.x_train.shape[1], self.y_train.shape[1])
        mw_hat, mb_hat = initializeGradients(self.num_total_layers, self.neurons_in_hidden_layers, self.x_train.shape[1], self.y_train.shape[1])
        vw_hat, vb_hat = initializeGradients(self.num_total_layers, self.neurons_in_hidden_layers, self.x_train.shape[1], self.y_train.shape[1])

        # run over the epochs
        for epoch in range(self.epochs):
            # run the algorithm to find the weights and biases in the epoch
            self.weights, self.bias, vw, vb, mw, mb, mw_hat, mb_hat, vw_hat, vb_hat = OptimizationFunctions.adam_gradient_descent(self, vw, vb, mw, mb, mw_hat, mb_hat, vw_hat, vb_hat, beta1, beta2, epsilon, batch_size, epoch)

            if bestConfigTestAccuracyRun == False:
                # find accuracy and loss
                trainingAccuracyCurrentEpoch, trainingLossCurrentEpoch = calculateTrainingAccuracyandLoss(self.x_train, self.y_train, self.weights, self.bias, self.num_total_layers, self.activation_function, self.loss_function)
                validationAccuracyCurrentEpoch, validationLossCurrentEpoch = calculateValidationAccuracyandLoss(self.x_validation, self.y_validation, self.weights, self.bias, self.num_total_layers, self.activation_function, self.loss_function)

                if mnistTestRun == False:
                    # log the accuracy and loss to wandb
                    printAndLogAccuracyAndLoss(epoch, trainingAccuracyCurrentEpoch, trainingLossCurrentEpoch, validationAccuracyCurrentEpoch, validationLossCurrentEpoch)

                if mnistTestRun == True and epoch == self.epochs - 1:
                    # print the accuracy for mnist dataset (question 10)
                    printAccuracyForMnistTest(trainingAccuracyCurrentEpoch, validationAccuracyCurrentEpoch)
        
        return self.weights, self.bias
    

    def runNadam(self, beta1, beta2, epsilon, batch_size, bestConfigTestAccuracyRun, mnistTestRun):
        '''
        Parameters:
            self : the self object of the class
            beta1 : beta1 value for the model
            beta2 : beta2 value for the model
            epsilon : epsilon value for the model
            batch_size : batch size used for the model
            bestConfigTestAccuracyRun : a boolean representing whether this model is used for finding test accuracy of the best model (Question 7)
            mnistTestRun : a boolean representing whether this model is used for finding the performance on Mnist dataset (Question 10)
        Returns:
            weights : weights learned
            bias : biases learned
        Function:
            runs the nadam gradient descent algorithm
        '''

        # initialize the parameters for the nadam algorithm
        vw, vb = initializeGradients(self.num_total_layers, self.neurons_in_hidden_layers, self.x_train.shape[1], self.y_train.shape[1])
        mw, mb = initializeGradients(self.num_total_layers, self.neurons_in_hidden_layers, self.x_train.shape[1], self.y_train.shape[1])
        mw_hat, mb_hat = initializeGradients(self.num_total_layers, self.neurons_in_hidden_layers, self.x_train.shape[1], self.y_train.shape[1])
        vw_hat, vb_hat = initializeGradients(self.num_total_layers, self.neurons_in_hidden_layers, self.x_train.shape[1], self.y_train.shape[1])

        # run over the epochs
        for epoch in range(self.epochs):
            # run the algorithm to find the weights and biases in the epoch
            self.weights, self.bias, vw, vb, mw, mb, mw_hat, mb_hat, vw_hat, vb_hat = OptimizationFunctions.nadam_gradient_descent(self, vw, vb, mw, mb, mw_hat, mb_hat, vw_hat, vb_hat, beta1, beta2, epsilon, batch_size)

            if bestConfigTestAccuracyRun == False:
                # find accuracy and loss
                trainingAccuracyCurrentEpoch, trainingLossCurrentEpoch = calculateTrainingAccuracyandLoss(self.x_train, self.y_train, self.weights, self.bias, self.num_total_layers, self.activation_function, self.loss_function)
                validationAccuracyCurrentEpoch, validationLossCurrentEpoch = calculateValidationAccuracyandLoss(self.x_validation, self.y_validation, self.weights, self.bias, self.num_total_layers, self.activation_function, self.loss_function)

                if mnistTestRun == False:
                    # log the accuracy and loss to wandb
                    printAndLogAccuracyAndLoss(epoch, trainingAccuracyCurrentEpoch, trainingLossCurrentEpoch, validationAccuracyCurrentEpoch, validationLossCurrentEpoch)
                
                if mnistTestRun == True and epoch == self.epochs - 1:
                    # print the accuracy for mnist dataset (question 10)
                    printAccuracyForMnistTest(trainingAccuracyCurrentEpoch, validationAccuracyCurrentEpoch)
        
        return self.weights, self.bias


    def fitModel(self, beta, beta1, beta2, epsilon, optimizer, batch_size, bestConfigTestAccuracyRun = False, mnistTestRun = False, trainPy = False, momentum = 0.0):
        '''
        Parameters:
            self : the self object of the class
            beta : beta value for the model
            beta1 : beta1 value for the model
            beta2 : beta2 value for the model
            epsilon : epsilon value for the model
            optimizer : optimization function for the model
            batch_size : batch size used for the model
            bestConfigTestAccuracyRun : a boolean representing whether this model is used for finding test accuracy of the best model (Question 7) which is by default set to false
            mnistTestRun : a boolean representing whether this model is used for finding the performance on Mnist dataset (Question 10) which is by default set to false
            momentum : momentum value for momentum and nesterov algorithms (which is by deafult set to 0)
        Returns:
            weights : weights learned
            bias : biases learned
        Function:
            driver function to run the algorithm
        '''

        if bestConfigTestAccuracyRun == False and mnistTestRun == False:
            # create a wandb run name with the hyperparameters of the model
            run = ""
            if trainPy == False:
                run = "LR_{}_OP_{}_EP_{}_BS_{}_INIT_{}_HL_{}_NHL_{}_AC_{}_WD_{}".format(self.learning_rate, optimizer, self.epochs, batch_size, self.initialization_method, self.num_hidden_layers, self.neurons_in_hidden_layers, self.activation_function, self.weight_decay)
            else:
                # include the loss type also if this model is used for the train.py implementation
                run = "LOSS_{}_LR_{}_OP_{}_EP_{}_BS_{}_INIT_{}_HL_{}_NHL_{}_AC_{}_WD_{}".format(self.loss_function, self.learning_rate, optimizer, self.epochs, batch_size, self.initialization_method, self.num_hidden_layers, self.neurons_in_hidden_layers, self.activation_function, self.weight_decay)
            print("current run : {}".format(run))
            wandb.run.name=run

        # call the corresponding runner function based on the choice of the optimization algorithm
        if(optimizer == "stochastic"):
            _, _ = self.runStochastic(batch_size, bestConfigTestAccuracyRun, mnistTestRun)
            
        elif(optimizer == "momentum"):
            if trainPy == False:
                _, _ = self.runMomentum(beta, batch_size, bestConfigTestAccuracyRun, mnistTestRun)
            else:
                _, _ = self.runMomentum(momentum, batch_size, bestConfigTestAccuracyRun, mnistTestRun)

        elif(optimizer == "nesterov"):
            if trainPy == False:
                _, _ = self.runNesterov(beta,batch_size, bestConfigTestAccuracyRun, mnistTestRun)
            else:
                _, _ = self.runNesterov(momentum,batch_size, bestConfigTestAccuracyRun, mnistTestRun)
        
        elif(optimizer == "rmsprop"):
            _, _ = self.runRmsprop(beta,epsilon,batch_size, bestConfigTestAccuracyRun, mnistTestRun)
        
        elif(optimizer == "adam"):
            _, _ = self.runAdam(beta1, beta2, epsilon, batch_size, bestConfigTestAccuracyRun, mnistTestRun)
        
        elif(optimizer == "nadam"):
            _, _ = self.runNadam(beta1, beta2, epsilon, batch_size, bestConfigTestAccuracyRun, mnistTestRun)

        # return the weights and biases to further calculate accuracy and loss for test data
        if bestConfigTestAccuracyRun == True or mnistTestRun == True or trainPy == True:
            return self.weights, self.bias
