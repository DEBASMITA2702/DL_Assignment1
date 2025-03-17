import wandb
import numpy as np
from matplotlib import pyplot as plt
from keras.datasets import fashion_mnist
from LoadDataset import loadAndSplitDataset

wandb.login()

#loads dataset from fashion_mnist
x_train, y_train, _, _, _, _ = loadAndSplitDataset(fashion_mnist)

#gives labels of the 10 output classes
output_classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

#creating subplots (2 rows, 5 columns)
fig, axes = plt.subplots(2, 5, figsize=(20, 6))
axes = axes.flatten()

#prepare a list to store the images
wandb_images = []

'''loop over each of the 10 classes'''
for current_class in range(10):
    '''get the first index in y_train that matches the specific current class'''
    temp_array = (y_train == current_class)
    first_match_index = np.argmax(temp_array)
    
    '''display the image in the subplot'''
    axes[current_class].imshow(x_train[first_match_index], cmap="gray")
    axes[current_class].set_title(output_classes[current_class])
    
    '''create a wandb image object'''
    single_image = wandb.Image(x_train[first_match_index], caption=[output_classes[current_class]])
    wandb_images.append(single_image)

#log all the images into wandb
wandb.init(project="Debasmita-DA6410-Assignment-1", name="Image for Question 1")
wandb.log({"Question 1": wandb_images})
wandb.finish()
