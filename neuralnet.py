import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation

from featureextraction import extract_features
import os

def get_data_files(searchdirectory, extension = ".csv"):
    targetfiles = []
    for root, dirs, files in os.walk(searchdirectory):
        for file in files:
            if extension in file:
                targetfiles.append(root+"/"+file)
    return targetfiles


if __name__ == '__main__':
    datafiles = get_data_files("testdata/traindatamix/Garbage")
    junkfeaturevectors = extract_features(datafiles)
    datafiles = get_data_files("testdata/traindatamix/Useful")
    usefulfeaturevectors = extract_features(datafiles)
    joined = np.c_[junkfeaturevectors,usefulfeaturevectors]
    joined = joined.transpose()
    # for a single-input model with 2 classes (binary):
    
    model = Sequential()
    model.add(Dense(1, input_dim=len(junkfeaturevectors), activation='sigmoid'))
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # generate dummy data

    dummydata = np.random.random((1000, 784))
    data = joined
    labels = np.random.randint(2, size=(1000, 1))
    labels = np.array([np.array([0]),np.array([0]),np.array([0]),np.array([0]),np.array([0]),np.array([0]),np.array([0]),np.array([0]),np.array([0]),np.array([0]),np.array([0]),np.array([0]),np.array([0]),np.array([0]),np.array([0]),np.array([0]),np.array([1]),np.array([1]),np.array([1]),np.array([1]),np.array([1]),np.array([1]),np.array([1]),np.array([1]),np.array([1]),np.array([1]),np.array([1]),np.array([1]),np.array([1]),np.array([1]),np.array([1]),np.array([1])])
    # train the model, iterating on the data in batches
    # of 32 samples
    model.fit(data, labels, nb_epoch=10, batch_size=32)
