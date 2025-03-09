import Utilities

def loadAndSplitDataset(dataset):
    '''loads the dataset to get the training and test data'''
    (x_train, y_train), (x_test, y_test) = dataset.load_data()

    '''
        generate random indices from the training set to generate the validation set
        training data is split into 10% validation and 90% training
    '''
    percentage = 0.1
    training_dataset_size = Utilities.getShape0(x_train)
    validation_dataset_size = int(percentage * training_dataset_size)
    validation_indices_in_training_dataset = Utilities.generateRandom(training_dataset_size, validation_dataset_size, False)

    '''validation dataset is created from the training dataset'''
    x_val = x_train[validation_indices_in_training_dataset]
    y_val = y_train[validation_indices_in_training_dataset]

    '''the data that went into validation dataset is now deleted from training dataset to create the new training dataset'''
    x_train = Utilities.deleteData(x_train, validation_indices_in_training_dataset)
    y_train = Utilities.deleteData(y_train, validation_indices_in_training_dataset)

    '''return the created datasets to the caller function'''
    return x_train, y_train, x_val, y_val, x_test, y_test