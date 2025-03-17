import Utilities

'''function to initialize the weights'''
def initializeWeight(initialization_method, number_total_layers, neurons_in_hidden_layer, length_input_data, length_output_data):
    '''
      Parameters:
        initialization_method : initailization method for the model
        number_total_layers : total number of layers for the model
        neurons_in_hidden_layer : number of neurons in the hidden layer of the model
        length_input_data : length of input data
        length_output_data : length of output data
      Returns:
        weights : a dictionary containing the weights across all the layers
      Function:
        initializes the weights according to the initlization method
    '''

    # a dictionary to represent the weights
    weights = {}

    # weight initialization for random initialization method
    if initialization_method == "random":
      # input layer (dimension of weight matrix : hidden x input)
      weights[1] = Utilities.random2D(neurons_in_hidden_layer, length_input_data)
      # hidden layers (dimension of weight matrix : hidden x hidden)
      for i in range(2, number_total_layers - 1):
        weights[i] = Utilities.random2DFloat(neurons_in_hidden_layer, neurons_in_hidden_layer)
      # output layer (dimension of weight matrix : output x hidden)
      weights[number_total_layers - 1] = Utilities.random2DFloat(length_output_data, neurons_in_hidden_layer)

    # weight initialization for xavier normal initialization method
    elif initialization_method == "xavier_nor":
      # input layer (dimension of weight matrix : hidden x input)
      weights[1] = Utilities.random2D(neurons_in_hidden_layer, length_input_data) * Utilities.xav_nor(length_input_data, neurons_in_hidden_layer)
      # hidden layers (dimension of weight matrix : hidden x hidden)
      for i in range(2, number_total_layers - 1):
        weights[i] = Utilities.random2D(neurons_in_hidden_layer, neurons_in_hidden_layer) * Utilities.xav_nor(neurons_in_hidden_layer, neurons_in_hidden_layer)
      # output layer (dimension of weight matrix : output x hidden)
      weights[number_total_layers - 1] = Utilities.random2D(length_output_data, neurons_in_hidden_layer) * Utilities.xav_nor(length_output_data, neurons_in_hidden_layer)

    # weight initialization for xavier uniform initialization method
    elif initialization_method == "xavier_uni":
      # input layer (dimension of weight matrix : hidden x input)
      weights[1] = Utilities.random2D(neurons_in_hidden_layer, length_input_data) * Utilities.xav_uni(length_input_data, neurons_in_hidden_layer)
      # hidden layers (dimension of weight matrix : hidden x hidden)
      for i in range(2, number_total_layers - 1):
        weights[i] = Utilities.random2D(neurons_in_hidden_layer, neurons_in_hidden_layer) * Utilities.xav_uni(neurons_in_hidden_layer, neurons_in_hidden_layer)
      # output layer (dimension of weight matrix : output x hidden)
      weights[number_total_layers - 1] = Utilities.random2D(length_output_data, neurons_in_hidden_layer) * Utilities.xav_uni(length_output_data, neurons_in_hidden_layer)

    # weight initialization for he normal initialization method
    elif initialization_method == "he_nor":
      # input layer (dimension of weight matrix : hidden x input)
      weights[1] = Utilities.random2D(neurons_in_hidden_layer, length_input_data) * Utilities.he_nor(length_input_data)
      # hidden layers (dimension of weight matrix : hidden x hidden)
      for i in range(2, number_total_layers - 1):
        weights[i] = Utilities.random2D(neurons_in_hidden_layer, neurons_in_hidden_layer) * Utilities.he_nor(neurons_in_hidden_layer)
      # output layer (dimension of weight matrix : output x hidden)
      weights[number_total_layers - 1] = Utilities.random2D(length_output_data, neurons_in_hidden_layer) * Utilities.he_nor(neurons_in_hidden_layer)

    # weight initialization for he uniform initialization method
    elif initialization_method == "he_uni":
      # input layer (dimension of weight matrix : hidden x input)
      weights[1] = Utilities.random2D(neurons_in_hidden_layer, length_input_data) * Utilities.he_uni(length_input_data)
      # hidden layers (dimension of weight matrix : hidden x hidden)
      for i in range(2, number_total_layers - 1):
        weights[i] = Utilities.random2D(neurons_in_hidden_layer, neurons_in_hidden_layer) * Utilities.he_uni(neurons_in_hidden_layer)
      # output layer (dimension of weight matrix : output x hidden)
      weights[number_total_layers - 1] = Utilities.random2D(length_output_data, neurons_in_hidden_layer) * Utilities.he_uni(neurons_in_hidden_layer)

    return weights



'''function to initialize the bias'''
def initializeBias(initialization_method, number_total_layers, neurons_in_hidden_layer, length_output_data):
    '''
      Parameters:
        initialization_method : initailization method for the model
        number_total_layers : total number of layers for the model
        neurons_in_hidden_layer : number of neurons in the hidden layer of the model
        length_input_data : length of input data
        length_output_data : length of output data
      Returns:
        bias : a dictionary containing the biases across all the layers
      Function:
        initializes the biases according to the initlization method
    '''

    # a dictionary to represent the biases
    bias = {}

    # bias initialization for random initialization method
    if initialization_method == "random":
      # input and hidden layers
      for i in range(1, number_total_layers - 1):
        bias[i] = Utilities.random1DFloat(neurons_in_hidden_layer)
      # output layer
      bias[number_total_layers - 1] = Utilities.random1DFloat(length_output_data)

    else:
      # input and hidden layers
      for i in range(1, number_total_layers - 1):
        bias[i] = Utilities.random1DFloat(neurons_in_hidden_layer)
      # output layer
      bias[number_total_layers - 1] = Utilities.random1DFloat(length_output_data)

    return bias


'''function to initialize the gradients'''
def initializeGradients(num_total_layers, neurons_in_hidden_layer, input_size, output_size):
    '''
      Parameters:
        number_total_layers : total number of layers for the model
        neurons_in_hidden_layer : number of neurons in the hidden layer of the model
        input_size : length of input data
        output_size : length of output data
      Returns:
        derivative_weights : a dictionary containing the initial weight change across all the layers
        derivative_biases : a dictionary containing the initial bias change across all the layers
      Function:
        initializes the change of weights and biases
    '''

    derivative_weights = {}
    # input layer
    derivative_weights[1] = Utilities.create_zeros2D(neurons_in_hidden_layer, input_size)
    # hidden layer
    for layer in range(2, num_total_layers - 1):
      derivative_weights[layer] = Utilities.create_zeros2D(neurons_in_hidden_layer, neurons_in_hidden_layer)
    # output layer
    derivative_weights[num_total_layers - 1] = Utilities.create_zeros2D(output_size, neurons_in_hidden_layer)

    derivative_biases = {}
    # input and hidden layer
    for layer in range(1, num_total_layers - 1):
      derivative_biases[layer] = Utilities.create_zeros1D(neurons_in_hidden_layer)
    # output layer
    derivative_biases[num_total_layers - 1] = Utilities.create_zeros1D(output_size)

    return derivative_weights, derivative_biases