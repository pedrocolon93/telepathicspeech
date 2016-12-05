import csv

import numpy
import numpy as np
from sklearn.decomposition import FastICA


def extract_features(filelist):
    S = []
    ica = FastICA(n_components=185)
    X1 = []
    for file in filelist:
        data = []
        with open(file, 'rb') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            for row in spamreader:
                conv = float(row[0])
                #t =numpy.array([conv])
                #t = [conv]
                data.append(conv)

        data = numpy.array(data)
        X1.append(data)

    t = X1[0]
    first = True
    for element in X1:
        if first:
            first = False
            continue
        t = np.c_[t,element]
    for index in range(0,len(t)):
        t[index] = numpy.array(t[index])
    X1 = numpy.array(t)
    X1 /= X1.std(axis=0)
    S_ = ica.fit_transform(X1)
    A_ = ica.mixing_

    return ica.mixing_

if __name__ == '__main__':
    featurevector = extract_features()

