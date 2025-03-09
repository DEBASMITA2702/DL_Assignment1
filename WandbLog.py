import wandb

def printAndLogAccuracyAndLoss(epoch, trainingAccuracy, trainingLoss, validationAccuracy, validationLoss):
    print("\n")
    print("Epoch = {}".format(epoch + 1))
    print("Training Accuracy = {}".format(trainingAccuracy))
    print("Validation Accuracy = {}".format(validationAccuracy))
    print("Training Loss = {}".format(trainingLoss))
    print("Validation Loss = {}".format(validationLoss))
    wandb.log({"training_accuracy" : trainingAccuracy, "validation_accuracy" : validationAccuracy, "training_loss" : trainingLoss, "validation_loss" : validationLoss, "epoch" : (epoch + 1)})