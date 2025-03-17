import wandb

def printAndLogAccuracyAndLoss(epoch, trainingAccuracy, trainingLoss, validationAccuracy, validationLoss):
    '''
      Parameters:
        epoch : curretn epoch number
        trainingAccuracy : training accuracy calculated
        trainingLoss : training loss calculated
        validationAccuracy : validation accuracy calculated
        validationLoss : validation loss calculated
      Returns:
        none
      Function:
        prints the accuracy and loss to terminal and logs the same to wandb
    '''

    print("\n")
    print("Epoch = {}".format(epoch + 1))
    print("Training Accuracy = {}".format(trainingAccuracy))
    print("Validation Accuracy = {}".format(validationAccuracy))
    print("Training Loss = {}".format(trainingLoss))
    print("Validation Loss = {}".format(validationLoss))
    wandb.log({"training_accuracy" : trainingAccuracy, "validation_accuracy" : validationAccuracy, "training_loss" : trainingLoss, "validation_loss" : validationLoss, "epoch" : (epoch + 1)})


def printAccuracyForMnistTest(trainingAccuracy, validationAccuracy):
    '''
      Parameters:
        trainingAccuracy : training accuracy calculated
        validationAccuracy : validation accuracy calculated
      Returns:
        none
      Function:
        prints the accuracy to terminal
    '''

    print("\nTraining Accuracy for Mnist dataset is : ", trainingAccuracy)
    print("Validation Accuracy for Mnist dataset is : ", validationAccuracy)