import numpy as np

def crossEntropyLoss(trueOutput, predictedOutput):
    output_log_value = np.log(predictedOutput + 1e-9)

    loss_value = -np.sum(trueOutput * output_log_value)
    
    return loss_value

def meanSquaredLoss(trueOutput, predictedOutput):
    difference = trueOutput - predictedOutput
    squared_difference = difference**2
    sum_value = np.sum(squared_difference)

    dim = trueOutput.shape[0]
    
    loss_value = sum_value / dim

    return loss_value