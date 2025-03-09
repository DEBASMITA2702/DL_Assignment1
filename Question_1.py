import wandb
import numpy as np
from matplotlib import pyplot as plt
from keras.datasets import fashion_mnist
from LoadDataset import loadAndSplitDataset

'''login to wandb to generate plot'''

'''loads dataset from fashion_mnist'''
x_train, y_train, _, _, _, _ = loadAndSplitDataset(fashion_mnist)

'''gives labels of the 10 output classes'''
output_class = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

'''initialize the project'''

_, getPlot = plt.subplots(2, 5, figsize = (20, 6))
getPlot = getPlot.flatten()
output_images = list()

for current_class in range(10):
  '''get the index where the image of the particular class is present'''
  image_class = np.argmax(y_train == current_class)

  '''get the image pixels'''
  getPlot[current_class].imshow(x_train[image_class],cmap = "gray")
  getPlot[current_class].set_title(output_class[current_class])

  '''generate plot for each class'''
  image = wandb.Image(x_train[image_class], caption = [output_class[current_class]])
  output_images.append(image)

'''log the images into wandb'''
wandb.login()
wandb.init(project = "Debasmita-DA6410-Assignment-1", name = "Image for Question 1")
wandb.log({"Question 1" : output_images})
wandb.finish()