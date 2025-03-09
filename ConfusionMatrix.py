import wandb
import numpy as np

def createAndPlotConfusionMatrix(true_values, predicted_values, dataset):
    output_class_tags = list()
    if dataset == "fashion_mnist":
        output_class_tags = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    else:
        output_class_tags = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]
    
    true_class_labels = list()
    predicted_class_labels = list()

    '''get class labels for the true outputs and their corresponding predictions'''
    data_size = true_values.shape[0]
    for point in range(data_size):
        true_class_index = np.argmax(true_values[point])
        true_class_labels.append(output_class_tags[true_class_index])
        
        predicted_class_index = np.argmax(predicted_values[point])
        predicted_class_labels.append(output_class_tags[predicted_class_index])

    '''plot the confusion matrix using sklearn'''
    wandb.sklearn.plot_confusion_matrix(true_class_labels, predicted_class_labels, output_class_tags)