import wandb
import numpy as np

''' this function is intended to create a confusion matrix for the test predictions and plot the same to wandb'''

def createAndPlotConfusionMatrix(true_values, predicted_values, dataset):
    '''
      Parameters:
        true_values : true outputs represented as one hot vectors
        predicted_values : predicted values from the output of softmax function
        dataset : choice of dataset that is being used
      Returns:
        none
      Function:
        creates a confusion matrix and plots it to wandb
    '''

    # create the output class labels based on the dataset
    output_class_tags = list()
    if dataset == "fashion_mnist":
        output_class_tags = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    else:
        output_class_tags = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]
    
    true_class_labels = list()
    predicted_class_labels = list()

    data_size = true_values.shape[0]

    # for each datapoint in the dataset, find the class labels for the true and predicted classes and store them
    for point in range(data_size):
        true_class_index = np.argmax(true_values[point])
        true_class_labels.append(output_class_tags[true_class_index])
        
        predicted_class_index = np.argmax(predicted_values[point])
        predicted_class_labels.append(output_class_tags[predicted_class_index])

    # plot the confusion matrix from the class labels identified above
    wandb.sklearn.plot_confusion_matrix(true_class_labels, predicted_class_labels, output_class_tags)