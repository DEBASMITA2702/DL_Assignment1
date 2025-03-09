import wandb
import DataPreProcessing
import WeightsAndBiasesAndGradientsInitialization
from CalculateAccuracyAndLoss import calculateTrainingAccuracyandLoss, calculateValidationAccuracyandLoss
from WeightsAndBiasesAndGradientsInitialization import initializeGradients
import OptimizationFunctions
import UpdateWeightsAndBiases
from WandbLog import printAndLogAccuracyAndLoss

class NeuralNetwork:
    def __init__(self, x_train, y_train, x_validation, y_validation, epochs, num_hidden_layers, neurons_in_hidden_layer, initialization_method, activation_function, loss_function, learning_rate, weight_decay):
        self.epochs = epochs
        self.num_hidden_layers = num_hidden_layers
        self.neurons_in_hidden_layers = neurons_in_hidden_layer
        self.initialization_method = initialization_method
        self.activation_function = activation_function
        self.loss_function = loss_function
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_total_layers = self.num_hidden_layers + 2


        self.x_train = DataPreProcessing.flattenAndNormalizeInputImage(x_train)
        self.y_train = DataPreProcessing.oneHotEncodeOutput(y_train)
        self.x_validation = DataPreProcessing.flattenAndNormalizeInputImage(x_validation)
        self.y_validation = DataPreProcessing.oneHotEncodeOutput(y_validation)


        input_length_for_weight_and_bias_initialization = self.x_train.shape[1]
        output_length_for_weight_and_bias_initialization = self.y_train.shape[1]
        self.weights = WeightsAndBiasesAndGradientsInitialization.initializeWeight(self.initialization_method, self.num_total_layers, 
                                                                       self.neurons_in_hidden_layers, 
                                                                       input_length_for_weight_and_bias_initialization, 
                                                                       output_length_for_weight_and_bias_initialization)
        self.bias = WeightsAndBiasesAndGradientsInitialization.initializeBias(self.initialization_method, self.num_total_layers, 
                                                                  self.neurons_in_hidden_layers,
                                                                  output_length_for_weight_and_bias_initialization)
    

    def runStochastic(self, batch_size):
        for epoch in range(self.epochs):
            self.weights, self.bias = OptimizationFunctions.stochastic_gradient_descent(self, batch_size)

            trainingAccuracyCurrentEpoch, trainingLossCurrentEpoch = calculateTrainingAccuracyandLoss(self.x_train, self.y_train, self.weights, self.bias, self.num_total_layers, self.activation_function, self.loss_function)
            validationAccuracyCurrentEpoch, validationLossCurrentEpoch = calculateValidationAccuracyandLoss(self.x_validation, self.y_validation, self.weights, self.bias, self.num_total_layers, self.activation_function, self.loss_function)

            printAndLogAccuracyAndLoss(epoch, trainingAccuracyCurrentEpoch, trainingLossCurrentEpoch, validationAccuracyCurrentEpoch, validationLossCurrentEpoch)

        return self.weights, self.bias


    def runMomentum(self, beta, batch_size):
        history_weights, history_biases = initializeGradients(self.num_total_layers, self.neurons_in_hidden_layers, self.x_train.shape[1], self.y_train.shape[1])

        for epoch in range(self.epochs):
            self.weights, self.bias, history_weights, history_biases = OptimizationFunctions.momentum_based_gradient_descent(self, history_weights, history_biases, beta, batch_size)

            trainingAccuracyCurrentEpoch, trainingLossCurrentEpoch = calculateTrainingAccuracyandLoss(self.x_train, self.y_train, self.weights, self.bias, self.num_total_layers, self.activation_function, self.loss_function)
            validationAccuracyCurrentEpoch, validationLossCurrentEpoch = calculateValidationAccuracyandLoss(self.x_validation, self.y_validation, self.weights, self.bias, self.num_total_layers, self.activation_function, self.loss_function)

            printAndLogAccuracyAndLoss(epoch, trainingAccuracyCurrentEpoch, trainingLossCurrentEpoch, validationAccuracyCurrentEpoch, validationLossCurrentEpoch)
        
        return self.weights, self.bias
    

    def runNesterov(self, beta, batch_size):
        history_weights, history_biases = initializeGradients(self.num_total_layers, self.neurons_in_hidden_layers, self.x_train.shape[1], self.y_train.shape[1])

        for epoch in range(self.epochs):
            uw, ub=dict(),dict()
            for i in range(1, len(history_weights)):
                uw[i] = beta * history_weights[i]
                ub[i] = beta * history_biases[i]

            inital_weights, inital_biases = UpdateWeightsAndBiases.momentum_update(self.weights, self.bias, uw, ub, self.learning_rate, self.weight_decay)

            self.weights, self.bias, history_weights, history_biases = OptimizationFunctions.nesterov_gradient_descent(self, uw, ub, history_weights, history_biases, inital_weights, inital_biases, beta, batch_size)

            trainingAccuracyCurrentEpoch, trainingLossCurrentEpoch = calculateTrainingAccuracyandLoss(self.x_train, self.y_train, self.weights, self.bias, self.num_total_layers, self.activation_function, self.loss_function)
            validationAccuracyCurrentEpoch, validationLossCurrentEpoch = calculateValidationAccuracyandLoss(self.x_validation, self.y_validation, self.weights, self.bias, self.num_total_layers, self.activation_function, self.loss_function)

            printAndLogAccuracyAndLoss(epoch, trainingAccuracyCurrentEpoch, trainingLossCurrentEpoch, validationAccuracyCurrentEpoch, validationLossCurrentEpoch)

        return self.weights, self.bias
    

    def runRmsprop(self, beta, epsilon, batch_size):
        history_weights, history_biases = initializeGradients(self.num_total_layers, self.neurons_in_hidden_layers, self.x_train.shape[1], self.y_train.shape[1])

        for epoch in range(self.epochs):
            self.weights, self.bias, history_weights, history_biases = OptimizationFunctions.rmsprop_gradient_descent(self, history_weights, history_biases, beta, epsilon, batch_size)
            
            trainingAccuracyCurrentEpoch, trainingLossCurrentEpoch = calculateTrainingAccuracyandLoss(self.x_train, self.y_train, self.weights, self.bias, self.num_total_layers, self.activation_function, self.loss_function)
            validationAccuracyCurrentEpoch, validationLossCurrentEpoch = calculateValidationAccuracyandLoss(self.x_validation, self.y_validation, self.weights, self.bias, self.num_total_layers, self.activation_function, self.loss_function)

            printAndLogAccuracyAndLoss(epoch, trainingAccuracyCurrentEpoch, trainingLossCurrentEpoch, validationAccuracyCurrentEpoch, validationLossCurrentEpoch)

        return self.weights, self.bias
    

    def runAdam(self, beta1, beta2, epsilon, batch_size):
        vw, vb = initializeGradients(self.num_total_layers, self.neurons_in_hidden_layers, self.x_train.shape[1], self.y_train.shape[1])
        mw, mb = initializeGradients(self.num_total_layers, self.neurons_in_hidden_layers, self.x_train.shape[1], self.y_train.shape[1])
        mw_hat, mb_hat = initializeGradients(self.num_total_layers, self.neurons_in_hidden_layers, self.x_train.shape[1], self.y_train.shape[1])
        vw_hat, vb_hat = initializeGradients(self.num_total_layers, self.neurons_in_hidden_layers, self.x_train.shape[1], self.y_train.shape[1])

        for epoch in range(self.epochs):
            self.weights, self.bias, vw, vb, mw, mb, mw_hat, mb_hat, vw_hat, vb_hat = OptimizationFunctions.adam_gradient_descent(self, vw, vb, mw, mb, mw_hat, mb_hat, vw_hat, vb_hat, beta1, beta2, epsilon, batch_size, epoch)

            trainingAccuracyCurrentEpoch, trainingLossCurrentEpoch = calculateTrainingAccuracyandLoss(self.x_train, self.y_train, self.weights, self.bias, self.num_total_layers, self.activation_function, self.loss_function)
            validationAccuracyCurrentEpoch, validationLossCurrentEpoch = calculateValidationAccuracyandLoss(self.x_validation, self.y_validation, self.weights, self.bias, self.num_total_layers, self.activation_function, self.loss_function)

            printAndLogAccuracyAndLoss(epoch, trainingAccuracyCurrentEpoch, trainingLossCurrentEpoch, validationAccuracyCurrentEpoch, validationLossCurrentEpoch)
        
        return self.weights, self.bias
    

    def runNadam(self, beta1, beta2, epsilon, batch_size):
        vw, vb = initializeGradients(self.num_total_layers, self.neurons_in_hidden_layers, self.x_train.shape[1], self.y_train.shape[1])
        mw, mb = initializeGradients(self.num_total_layers, self.neurons_in_hidden_layers, self.x_train.shape[1], self.y_train.shape[1])
        mw_hat, mb_hat = initializeGradients(self.num_total_layers, self.neurons_in_hidden_layers, self.x_train.shape[1], self.y_train.shape[1])
        vw_hat, vb_hat = initializeGradients(self.num_total_layers, self.neurons_in_hidden_layers, self.x_train.shape[1], self.y_train.shape[1])

        for epoch in range(self.epochs):
            self.weights, self.bias, vw, vb, mw, mb, mw_hat, mb_hat, vw_hat, vb_hat = OptimizationFunctions.nadam_gradient_descent(self, vw, vb, mw, mb, mw_hat, mb_hat, vw_hat, vb_hat, beta1, beta2, epsilon, batch_size, epoch)

            trainingAccuracyCurrentEpoch, trainingLossCurrentEpoch = calculateTrainingAccuracyandLoss(self.x_train, self.y_train, self.weights, self.bias, self.num_total_layers, self.activation_function, self.loss_function)
            validationAccuracyCurrentEpoch, validationLossCurrentEpoch = calculateValidationAccuracyandLoss(self.x_validation, self.y_validation, self.weights, self.bias, self.num_total_layers, self.activation_function, self.loss_function)

            printAndLogAccuracyAndLoss(epoch, trainingAccuracyCurrentEpoch, trainingLossCurrentEpoch, validationAccuracyCurrentEpoch, validationLossCurrentEpoch)

        return self.weights, self.bias


    def fitModel(self, beta, beta1, beta2, epsilon, optimizer, batch_size):
        '''generates run name and logs it to wandb'''
        run="LR_{}_OP_{}_EP_{}_BS_{}_INIT_{}_HL_{}_NHL_{}_AC_{}_WD_{}".format(self.learning_rate, optimizer, self.epochs, batch_size, self.initialization_method, self.num_hidden_layers, self.neurons_in_hidden_layers, self.activation_function, self.weight_decay)
        print("run name = {}".format(run))
        wandb.run.name=run

        if(optimizer == "stochastic"):
            _, _ = self.runStochastic(batch_size)
            
        elif(optimizer == "momentum"):
            _, _ = self.runMomentum(beta, batch_size)

        elif(optimizer == "nesterov"):
            _, _ = self.runNesterov(beta,batch_size)
        
        elif(optimizer == "rmsprop"):
            _, _ = self.runRmsprop(beta,epsilon,batch_size)
        
        elif(optimizer == "adam"):
            _, _ = self.runAdam(beta1, beta2, epsilon, batch_size)
        
        elif(optimizer == "nadam"):
            _, _ = self.runNadam(beta1, beta2, epsilon, batch_size)