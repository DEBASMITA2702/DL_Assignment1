from Utilities import checkIfNone

def setValuesFromArgument(argument_passed):
    '''
    Parameters:
        argument_passed : arguments that are passed as part of command line
    Returns:
        hyperparam_dictionary : a dictionary with all the hyperparameter values set
    Function:
        extracts and sets the values of the hyperparameters (needed for running train.py)
    '''

    # default values of each of the hyperparameter according to the config of my best model obtained from sweeping
    set_project_name = "Debasmita-DA6410-Assignment-1"
    set_entity_name = "cs24m015-indian-institute-of-technology-madras"
    set_learning_rate = 0.005
    set_momentum = 0.999
    set_beta = 0.999
    set_beta1 = 0.9
    set_beta2 = 0.999
    set_epsilon = 1e-5
    set_optimizer = "adam"
    set_batch_size = 128
    set_loss_type = "cross_entropy"
    set_epochs = 10
    set_weight_decay = 0.0005
    set_dataset_name = "fashion_mnist"
    set_weight_initialization = "he_nor"
    set_hidden_layers = 3
    set_neurons_in_hidden = 128
    set_activation = "relu"
    set_confusion_matrix = 0
    set_test = 0

    # set the values if they are passed as arguments on command line
    if checkIfNone(argument_passed.wandb_project):
        set_project_name = argument_passed.wandb_project
    if checkIfNone(argument_passed.wandb_entity):
        set_entity_name = argument_passed.wandb_entity
    if checkIfNone(argument_passed.dataset):
        set_dataset_name = argument_passed.dataset
    if checkIfNone(argument_passed.learning_rate):
        set_learning_rate = argument_passed.learning_rate
    if checkIfNone(argument_passed.momentum):
        set_momentum = argument_passed.momentum
    if checkIfNone(argument_passed.beta):
        set_beta = argument_passed.beta
    if checkIfNone(argument_passed.beta1):
        set_beta1 = argument_passed.beta1
    if checkIfNone(argument_passed.beta2):
        set_beta2 = argument_passed.beta2
    if checkIfNone(argument_passed.epsilon):
        set_epsilon = argument_passed.epsilon
    if checkIfNone(argument_passed.optimizer):
        set_optimizer = argument_passed.optimizer
    if checkIfNone(argument_passed.batch_size):
        set_batch_size = argument_passed.batch_size
    if checkIfNone(argument_passed.loss):
        set_loss_type = argument_passed.loss
    if checkIfNone(argument_passed.epochs):
        set_epochs = argument_passed.epochs
    if checkIfNone(argument_passed.weight_decay):
        set_weight_decay = argument_passed.weight_decay
    if checkIfNone(argument_passed.weight_init):
        set_weight_initialization = argument_passed.weight_init
    if checkIfNone(argument_passed.num_layers):
        set_hidden_layers = argument_passed.num_layers
    if checkIfNone(argument_passed.hidden_size):
        set_neurons_in_hidden = argument_passed.hidden_size
    if checkIfNone(argument_passed.activation):
        set_activation = argument_passed.activation
    if checkIfNone(argument_passed.confusion):
        set_confusion_matrix = argument_passed.confusion
    if checkIfNone(argument_passed.test):
        set_test = argument_passed.test

    # create and set the hyperparameter dictionary
    hyperparam_dictionary = dict()
    
    hyperparam_dictionary["project_name"] = set_project_name
    hyperparam_dictionary["entity_name"] = set_entity_name
    hyperparam_dictionary["learning_rate"] = set_learning_rate
    hyperparam_dictionary["momentum"] = set_momentum
    hyperparam_dictionary["beta"] = set_beta
    hyperparam_dictionary["beta1"] = set_beta1
    hyperparam_dictionary["beta2"] = set_beta2
    hyperparam_dictionary["epsilon"] = set_epsilon
    hyperparam_dictionary["optimizer"] = set_optimizer
    hyperparam_dictionary["batch_size"] = set_batch_size
    hyperparam_dictionary["loss_type"] = set_loss_type
    hyperparam_dictionary["epochs"] = set_epochs
    hyperparam_dictionary["weight_decay"] = set_weight_decay
    hyperparam_dictionary["dataset_name"] = set_dataset_name
    hyperparam_dictionary["weight_initialization"] = set_weight_initialization
    hyperparam_dictionary["hidden_layers"] = set_hidden_layers
    hyperparam_dictionary["neurons_in_hidden"] = set_neurons_in_hidden
    hyperparam_dictionary["activation"] = set_activation
    hyperparam_dictionary["confusion_matrix"] = set_confusion_matrix
    hyperparam_dictionary["test"] = set_test

    return hyperparam_dictionary