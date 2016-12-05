import numpy as np
import matplotlib.pyplot as plt
from keras.engine import Merge
from scipy import signal
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, LSTM

from sklearn.decomposition import FastICA, PCA

# Generate sample data


n_samples = 2000
n_obs = 5000*2

basisfuncts = []
for i in range(0,n_obs):
    np.random.seed(i)
    time = np.linspace(0, 8, n_samples)

    s1 = np.sin(2 * time)  # Signal 1 : sinusoidal signal
    s2 = np.sign(np.sin(3 * time))  # Signal 2 : square signal
    s3 = signal.sawtooth(2 * np.pi * time)  # Signal 3: saw tooth signal

    S = np.c_[s1, s2, s3]
    S += 0.2 * np.random.normal(size=S.shape)  # Add noise

    S /= S.std(axis=0)  # Standardize data

    # Mix data
    A = np.array([[1, 1, 1], [0.5, 2, 1.0], [1.5, 1.0, 2.0]])  # Mixing matrix
    X = np.dot(S, A.T)  # Generate observations

    # Compute ICA
    ica = FastICA()
    S_ = ica.fit_transform(X)  # Reconstruct signals
    A_ = ica.mixing_  # Get estimated mixing matrix

    basisfuncts.append(S_)

# for a single-input model with 2 classes (binary):

# model = Sequential()

#
# model.add(Dense(185, input_dim=(2000,3), init='uniform', activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(100, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(25, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(1, activation='sigmoid'))

# model = Sequential()
# model.add(LSTM(185,input_shape=(n_samples,3,n_obs),activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(100, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(1, activation='relu'))
#model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
branches = []
tstlen = len(basisfuncts[0][0])
for i in range(0,len(basisfuncts[0][0])):
    dimbranch = Sequential()
    dimbranch.add(Dense(185, input_dim=2000))
    branches.append(dimbranch)


merged = Merge(branches, mode='concat')

final_model = Sequential()
final_model.add(merged)
final_model.add(Dense(1, activation='softmax'))



# two branch Sequential

final_model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy','precision','recall'])


# generate dummy data
# rdata = np.random.random((n_obs, n_samples))
tdata = (np.array(basisfuncts))
# data = np.vstack((tdata,rdata))
data = tdata
a = np.ones(n_obs*2)
a_ = []
for i in range(0,len(a)):
    # t = a[i]
    t = np.array([a[i]])
    a_.append(t)
a_ = np.array(a_)
b =np.zeros(n_obs)
b_ = []
for i in range(0,len(b)):
    # t = b[i]
    t = np.array([b[i]])
    b_.append(t)
    # b[i]= np.array([b[i]])
b_ = np.array(b_)
# labels = np.vstack((a_,b_))
labels = a_
# train the model, iterating on the data in batches
# of 32 samples
# model.fit(data, labels, nb_epoch=10, batch_size=64)
#
# final_model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
inputdataarray = []
for basisfunction in basisfuncts[0][0]:
    inputdataarray.append([])
for basisfunctioncollection in data:

    for basisfunctcoeffindex, basisfunctioncoefficient in enumerate(basisfunctioncollection):

        for componentindex, component in enumerate(basisfunctioncoefficient):
            try:
                inputdataarray[componentindex][basisfunctcoeffindex].append(component)
            except:
                inputdataarray[componentindex].append([])
                inputdataarray[componentindex][basisfunctcoeffindex].append(component)
        print ""





final_model.fit(inputdataarray, targets)
# score = model.evaluate(np.array(testdatabasisfuncts), a_, batch_size=16)
# print score
# print model.metrics_names
# a = np.ones(n_obs)
# a_ = []
# for i in range(0,len(a)):
#     # t = a[i]
#     t = np.array([a[i]])
#     a_.append(t)
# a_ = np.array(a_)
# score = model.evaluate(np.random.random((n_obs, n_samples)),b_, batch_size=16)
# print score
# print model.metrics_names
