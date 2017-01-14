# Kaggle FKD Tutorial

# rid deprecation warnings
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
    # Pass a list of cols if you're only interested in a subset of the
    # target columns.
    
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
net1 = NeuralNet(layers=[  
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
net1.fit(X, y)

def plot_sample(x, y, axis):
    img = x.reshape(96, 96)
    axis.imshow(img, cmap='gray')
    axis.scatter(y[0::2] * 48 + 48, y[1::2] * 48 + 48, marker='x', s=10)

X, _ = load(test=True)
y_pred = net1.predict(X)

fig = plt.figure(figsize=(6, 6))
fig.subplots_adjust(
    left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

for i in range(16):
    ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
    plot_sample(X[i], y_pred[i], ax)

plt.show()
