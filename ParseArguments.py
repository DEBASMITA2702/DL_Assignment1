import warnings
import argparse

warnings.filterwarnings("ignore")

def collectArguments():
    '''
    Parameters:
        none
    Returns:
        an object conataining all the values that are passed as part of command line argument
    Function:
        collects all the arguemnts passed as part of the command line
    '''

    argument_passed = argparse.ArgumentParser(description = 'Arguments for model hyperparameters')
    argument_passed.add_argument('-wp','--wandb_project', help = "Project name used to track experiments in Weights & Biases dashboard")
    argument_passed.add_argument('-we','--wandb_entity', help = "Wandb Entity used to track experiments in the Weights & Biases dashboard")
    argument_passed.add_argument('-d','--dataset', help = "Dataset to work on \n choices: ['mnist', 'fashion_mnist']")
    argument_passed.add_argument('-e','--epochs', type = int, help = "Number of epochs to train neural network")
    argument_passed.add_argument('-b','--batch_size', type = int, help = "Batch size used to train neural network")
    argument_passed.add_argument('-l','--loss', help = "Loss function to use for backpropagation \n choices: ['mean_squared_error', 'cross_entropy']")
    argument_passed.add_argument('-o','--optimizer', help = "Optimization algorithm to use for training the model \n choices: ['stochastic', 'momentum', 'nesterov', 'rmsprop', 'adam', 'nadam']")
    argument_passed.add_argument('-lr','--learning_rate', type = float, help = "Learning rate used to optimize model parameters")
    argument_passed.add_argument('-m','--momentum', type = float, help = "Momentum used by momentum and nag optimizers")
    argument_passed.add_argument('-beta','--beta', type = float, help = "Beta used by rmsprop optimizer")
    argument_passed.add_argument('-beta1','--beta1', type = float, help = "Beta1 used by adam and nadam optimizers")
    argument_passed.add_argument('-beta2','--beta2', type = float, help = "Beta2 used by adam and nadam optimizers")
    argument_passed.add_argument('-eps','--epsilon', type = float, help = "Epsilon used by optimizers")
    argument_passed.add_argument('-w_d','--weight_decay', type = float, help = "Weight decay used by optimizers")
    argument_passed.add_argument('-w_i','--weight_init', help = "Weight initialization technique to use \n choices: ['random', 'xavier_nor', 'xavier_uni', 'he_nor', 'he_uni']")
    argument_passed.add_argument('-nhl','--num_layers', type = int, help = "Number of hidden layers used in feedforward neural network")
    argument_passed.add_argument('-sz','--hidden_size', type = int, help = "Number of hidden neurons in a feedforward layer")
    argument_passed.add_argument('-a','--activation', help = "Activation function to use for the model \n choices: ['identity', 'sigmoid', 'tanh', 'relu']")
    argument_passed.add_argument('-c','--confusion', type = int, help = "Whether you want to generate confusion matrix or not \n choices: [0,1]")
    argument_passed.add_argument('-t','--test', type = int, help = "Whether you want to calculate and print test accuracy or not \n choices: [0,1]")

    return argument_passed.parse_args()