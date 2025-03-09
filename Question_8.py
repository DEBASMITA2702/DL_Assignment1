import wandb
from LoadDataset import loadAndSplitDataset
from FeedForwardNeuralNetwork import NeuralNetwork
from keras.datasets import fashion_mnist

'''login to wandb to generate plot'''
wandb.login()

def compareLossTypes():
    '''initialize to project adn create a config'''
    wandb.init(project = "Debasmita-DA6410-Assignment-1")
    config = wandb.config
   
    '''loads the dataset'''
    x_train, y_train, x_val, y_val, _, _ = loadAndSplitDataset(fashion_mnist)
    
    '''create an object of the FeedForwardNeuralNetwork class which has all the required functions
      pass the parameters to the constructor as a sweep value. this will change the values with each run of the sweep.
      call the fitting fucntion and pass the optimizer as a parameter
    '''
    Model_for_loss_testing = NeuralNetwork(x_train, y_train, x_val, y_val, epochs = config.number_of_epochs, 
                                  num_hidden_layers = config.number_of_hidden_layers, 
                                  neurons_in_hidden_layer = config.neurons_in_each_hidden_layers, 
                                  initialization_method = config.initialization_method, 
                                  activation_function = config.activation_function, 
                                  loss_function = config.loss_type, 
                                  learning_rate = config.learning_rate, 
                                  weight_decay = config.weight_decay)

    Model_for_loss_testing.fitModel(beta = config.beta_value, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-5, 
                                    optimizer = config.optimizer_technique, 
                                    batch_size = config.batch_size)


'''sweep configuration'''
configuration_values = {
    'method': 'bayes',
    'name': 'Cross Entropy vs Mean Squared',
    'metric': {
        'goal': 'maximize',
        'name': 'validation_accuracy'
    },
    'parameters': {
        'initialization_method': {'values': ['he_nor']},
        'number_of_hidden_layers' : {'values' : [3]},
        'neurons_in_each_hidden_layers' : {'values' : [128]},

        'learning_rate': {'values':[5e-3]},
        'beta_value' : {'values' : [0.999]},
        'optimizer_technique' : {'values' : ['adam']},

        'batch_size': {'values': [128]},
        'number_of_epochs': {'values': [10]},
        'loss_type' : {'values' : ['cross_entropy', 'mean_squared_error']},
        'activation_function' : {'values' : ['relu']},
        'weight_decay' : {'values' : [0.0005]}
    }
}

'''create a sweep id in the current project'''
sweep_agent_id = wandb.sweep(sweep = configuration_values, project = "Debasmita-DA6410-Assignment-1")

'''generate a sweep agent to run the sweep'''
wandb.agent(sweep_agent_id, function = compareLossTypes, count = 2)
wandb.finish()