# Kaggle Facial Keypoint Detection Tutorial

# deprecation warnings
def warn(*args, **kwargs):
    pass

import os
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.warn = warn

from pandas import read_csv
from sklearn.utils import shuffle
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet


FTEST  = '~/Documents/introML/introML/deep_learn/test.csv'
FTRAIN = '~/Documents/introML/introML/deep_learn/training.csv'


def load(test=False, cols=None):
    # Loads data from FTEST if file present, otherwise from FTRAIN.
    # Pass list of cols if only interested in subset of target columns.
    
    file_name = FTEST if test else FTRAIN
    # load pandas dataframe
    df = read_csv(os.path.expanduser(file_name))  

    # image column has pixel values separated by space; convert to numpy arrays
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))

    # get a subset of columns
    if cols:  
        df = df[list(cols) + ['Image']]

    print(df.count()) 
    # drop all rows twith missing values 
    df =  df.dropna()   

    # scale pixel values to [0, 1] as opposed to [0, 255]
    X = np.vstack(df['Image'].values) / 255.0  
    X = X.astype(np.float32)

    if not test:  # only FTRAIN has any target columns
        y = df[df.columns[:-1]].values
        y = (y - 48) / 48  # scale target coordinates to [-1, 1]
        X, y = shuffle(X, y, random_state=42)
        y = y.astype(np.float32)
    else:
        y = None

    return X, y

# describe data 
'''
print("X.shape == {}; X.min == {:.3f}; X.max == {:.3f}".format(
    X.shape, X.min(), X.max()))
print("y.shape == {}; y.min == {:.3f}; y.max == {:.3f}".format(
    y.shape, y.min(), y.max()))
'''

# first model
net = NeuralNet(layers=[  
                       ('input', layers.InputLayer),
                       ('hidden', layers.DenseLayer),
                       ('output', layers.DenseLayer),
                       ],

    # layer parameters:
    input_shape=(None, 9216),  # 96x96 input pixels per batch
    hidden_num_units=100,      # number of nodes in hidden layer
    output_nonlinearity=None,  # output layer uses identity function?
    output_num_units=30,       # 30 target values

    # optimization method:
    update=nesterov_momentum,
    update_learning_rate=0.01,
    update_momentum=0.9,

    regression=True,  # flag indicates regression problem, not classification
    max_epochs=100,   # number of trials
    verbose=1,
    )

X, y = load()

net.fit(X, y)

# wrapper for convolutional neural net
def load2d(test=False, cols=None):
    X, y = load(test=test)
    X = X.reshape(-1, 1, 96, 96)
    return X, y

cnet = NeuralNet(layers=[
                        ('input', layers.InputLayer),
                        ('conv1', layers.Conv2DLayer),
                        ('pool1', layers.MaxPool2DLayer),
                        ('conv2', layers.Conv2DLayer),
                        ('pool2', layers.MaxPool2DLayer),
                        ('conv3', layers.Conv2DLayer),
                        ('pool3', layers.MaxPool2DLayer),
                        ('hidden4', layers.DenseLayer),
                        ('hidden5', layers.DenseLayer),
                        ('output', layers.DenseLayer),
                        ],
    input_shape=(None, 1, 96, 96),
    conv1_num_filters=32, 
    conv1_filter_size=(3,3),
    pool1_pool_size=(2, 2),
    conv2_num_filters=64, 
    conv2_filter_size=(2, 2), 
    pool2_pool_size=(2, 2),
    conv3_num_filters=128, 
    conv3_filter_size=(2, 2), 
    pool3_pool_size=(2, 2),
    hidden4_num_units=500, 
    hidden5_num_units=500,
    output_num_units=30, 
    output_nonlinearity=None,

    update_learning_rate=0.01,
    update_momentum=0.9,

    regression=True,
    max_epochs=100,
    verbose=1,
    )


X, y = load2d()  # load 2-d data
cnet.fit(X, y)