import csv
import random

import numpy as np
import os
from keras.layers import Dense, Dropout
from keras.models import Sequential
from sklearn.decomposition import FastICA


def get_data_files(searchdirectory, extension = ".csv"):
    targetfiles = []
    for root, dirs, files in os.walk(searchdirectory):
        for file in files:
            if extension in file:
                targetfiles.append(root+"/"+file)
    return targetfiles


def get_observations(filelist):
    X1 = []
    for file in filelist:
        data = []
        with open(file, 'rb') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            for row in spamreader:
                conv = float(row[0])
                # t =numpy.array([conv])
                # t = [conv]
                data.append(conv)

        data = np.array(data)
        X1.append(data)

    t = X1[0]
    first = True
    for element in X1:
        if first:
            first = False
            continue
        t = np.c_[t, element]
    for index in range(0, len(t)):
        t[index] = np.array(t[index])

    return t


if __name__ == '__main__':

    #load truth observationss
    directory = "testdata/handdata"
    files = get_data_files(directory)
    t_observations = get_observations(files)
    observationsarray = []
    for observation in t_observations.T:
        observationsarray.append((observation,np.array([1])))

    #load false observations
    directory = "testdata/garbagedata"
    files = get_data_files(directory)
    f_observations = get_observations(files)

    for observation in f_observations.T:
       observationsarray.append((observation, np.array([0])))

    #join and mix

    random.shuffle(observationsarray)

    randomdata = []
    labels = []
    for observation in observationsarray:
        randomdata.append(observation[0])
        labels.append(observation[1])


    n_obs = len(randomdata)
    n_samples = len(randomdata[0])

    randomdata = np.array(randomdata)
    labels = np.array(labels)

    randomdata = randomdata.T
    # Compute ICA
    ica = FastICA()
    S_ = ica.fit_transform(randomdata)  # Reconstruct signals
    # A_ = ica.mixing_  # Get estimated mixing matrix

    model = Sequential()

    model.add(Dense(n_obs, input_dim=n_samples, init='uniform', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(40, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy','recall','precision', 'fmeasure'])

    from sklearn.metrics import confusion_matrix


    model.fit(S_.T,labels,nb_epoch=20)

    # load test observationss
    directory = "testdata/OriginalData"
    files = get_data_files(directory)
    t_observations = get_observations(files)
    ica = FastICA()
    S_ = ica.fit_transform(t_observations)
    evalables = np.array([np.array([0]),np.array([0]),np.array([0]),np.array([1]),np.array([1]),np.array([1]),np.array([1]),np.array([1]),np.array([1]),np.array([1]),np.array([1]),np.array([1]),np.array([1]),np.array([1]),np.array([1])])

    res = model.evaluate(S_.T, evalables)
    res2 = model.predict(S_.T)
    print "conf"
    print confusion_matrix(evalables[:1,],res2)
    print "end"
    print res
    print model.metrics_names

