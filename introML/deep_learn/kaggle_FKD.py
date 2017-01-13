import os
import numpy as np

from pandas import read_csv
from sklearn.utils import shuffle, validation
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet

FTRAIN = '~/Documents/introML/introML/deep_learn/training.csv'
FTEST = '~/Documents/introML/introML/deep_learn/training.csv'

def load(test=False, cols=None):
    ''' 
    loads data from FTEST if test is true, otherwise from FTRAIN.
    pass a list of cols if interested in subset of target columns. 
    '''
    
    file_name = FTEST if test else FTRAIN
    df = read_csv(os.path.expanduser(file_name)) # load pandas dataframe

    # The Image column has pixel values separated by space; convert
    # the values to numpy arrays:
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))

    if cols:  # get a subset of columns
        df = df[list(cols) + ['Image']]

    print(df.count())  # prints the number of values for each column
    df = df.dropna()  # drop all rows that have missing values in them

    X = np.vstack(df['Image'].values) / 255.  # scale pixel values to [0, 1]
    X = X.astype(np.float32)

    if not test:  # only FTRAIN has any target columns
        Y = df[df.columns[:-1]].values
        Y = (Y - 48) / 48  # scale target coordinates to [-1, 1]
        X = shuffle(X, Y, random_state=42)  # shuffle train data
        Y = shuffle(X, Y, random_state=42) 
        Y = Y.astype(np.float32)
    else:
        Y = None

    return X, Y

'''
X, Y = load()
print("X.shape == {}; X.min == {:.3f}; X.max == {:.3f}".format(
    X.shape, X.min(), X.max()))
print("y.shape == {}; y.min == {:.3f}; y.max == {:.3f}".format(
    Y.shape, Y.min(), Y.max()))
'''

net1 = NeuralNet(layers = [('input', layers.InputLayer),
                           ('hidden', layers.DenseLayer),
                           ('output', layers.DenseLayer)],
                           # layer parameters:
                           input_shaper = (None, 9216), # 96x96 pixels per batch
                           hidden_num_units = 100, # number of units in hidden layer
                           output_nonlinearity = None, 
                           output_num_units = 30, # 30 target values

                           # optimization method:
                           update = nesterov_momentum,
                           update_learning_rate = 0.1,
                           update_momentum = 0.9,

                           regression = True, # indicate regression problem
                           max_epochs = 300, 
                           verbose = 1,)

X, Y = load()
net1.fit(X, Y)