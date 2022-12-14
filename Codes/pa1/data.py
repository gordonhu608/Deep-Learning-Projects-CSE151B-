import numpy as np
import os

def load_data(train = True):
    """
    Load the data from disk

    Parameters
    ----------
    train : bool
        Load training data if true, else load test data
    Returns
    -------
        Tuple:
            Images
            Labels
    """
    directory = 'train' if train else 'test'
    patterns = np.load(os.path.join('./data/', directory, 'images.npz'))['arr_0']
    labels = np.load(os.path.join('./data/', directory, 'labels.npz'))['arr_0']
    return patterns.reshape(len(patterns), -1), labels

def z_score_normalize(X, u = None, xd = None):
    """
    Performs z-score normalization on X. 

    f(x) = (x - μ) / σ
        where 
            μ = mean of x
            σ = standard deviation of x

    Parameters
    ----------
    X : np.array
        The data to z-score normalize
    u (optional) : np.array
        The mean to use when normalizing
    sd (optional) : np.array
        The standard deviation to use when normalizing

    Returns
    -------
        Tuple:
            Transformed dataset with mean 0 and stdev 1
            Computed statistics (mean and stdev) for the dataset to undo z-scoring.
    """
 
    if u is None:
        u = np.mean(axis=0)
    if xd is None:
        xd = np.std(axis=0)
    u = np.expand_dims(u, axis=1)
    xd = np.expand_dims(xd, axis=1)
    return (X - np.repeat(u, X.shape[0], axis=1).T)/np.repeat(xd, X.shape[0], axis=1).T

def min_max_normalize(X, _min = None, _max = None):
    """
    Performs min-max normalization on X. 

    f(x) = (x - min(x)) / (max(x) - min(x))

    Parameters
    ----------
    X : np.array
        The data to min-max normalize
    _min (optional) : np.array
        The min to use when normalizing
    _max (optional) : np.array
        The max to use when normalizing

    Returns
    -------
        Tuple:
            Transformed dataset with all values in [0,1]
            Computed statistics (min and max) for the dataset to undo min-max normalization.
    """

    if _min is None:
        _minn = np.amin(X)
    else:
        _minn = _min
    if _max is None:
        _maxx = np.amax(X)
    else:
        _maxx = _max 
    
    #if _min is None:
    #    _minn = np.min(axis=0)
    #else:
    #    _minn = _min
    #if _max is None:
    #    _maxx = np.max(axis=0)
    #else:
    #    _maxx = _max
    #_minn = np.expand_dims(_minn, axis=1)
    #_maxx = np.expand_dims(_maxx, axis=1)
    #_minn = np.repeat(_minn, X.shape[0], axis=1).T
    #_maxx = np.repeat(_maxx, X.shape[0], axis=1).T
    res = (X - _minn) / (_maxx - _minn)
    return res
    
def onehot_encode(y):
    """
    Performs one-hot encoding on y.

    Ideas:
        NumPy's `eye` function

    Parameters
    ----------
    y : np.array
        1d array (length n) of targets (k)

    Returns
    -------
        2d array (shape n*k) with each row corresponding to a one-hot encoded version of the original value.
    """
    n_values = np.max(y) + 1
    print('a')
    return np.eye(n_values)[y]

def onehot_decode(y):
    """
    Performs one-hot decoding on y.

    Ideas:
        NumPy's `argmax` function 

    Parameters
    ----------
    y : np.array
        2d array (shape n*k) with each row corresponding to a one-hot encoded version of the original value.

    Returns
    -------
        1d array (length n) of targets (k)
    """
    return np.argmax(y, axis=1)

def shuffle(dataset):
    """
    Shuffle dataset.

    Make sure that corresponding images and labels are kept together. 
    Ideas: 
        NumPy array indexing 
            https://numpy.org/doc/stable/user/basics.indexing.html#advanced-indexing

    Parameters
    ----------
    dataset
        Tuple containing
            Images (X)
            Labels (y)

    Returns
    -------
        Tuple containing
            Images (X)
            Labels (y)
    """
    X, y = dataset
    
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    shuffled_X, shuffled_y = X[indices], y[indices]

    return shuffled_X, shuffled_y


def append_bias(X):
    """
    Append bias term for dataset.

    Parameters
    ----------
    X
        2d numpy array with shape (N,d)

    Returns
    -------
        2d numpy array with shape (N,(d+1))
    """
    return np.hstack([X, np.ones((X.shape[0],1))])

def generate_minibatches(dataset, batch_size=64):
    X, y = dataset
    l_idx, r_idx = 0, batch_size
    while r_idx < len(X):
        yield X[l_idx:r_idx], y[l_idx:r_idx]
        l_idx, r_idx = r_idx, r_idx + batch_size

    yield X[l_idx:], y[l_idx:]

def generate_k_fold_set(dataset, k = 10): 
    X, y = dataset

    order = np.random.permutation(len(X))
    
    fold_width = len(X) // k

    l_idx, r_idx = 0, fold_width

    for i in range(k):
        train = np.concatenate([X[order[:l_idx]], X[order[r_idx:]]]), np.concatenate([y[order[:l_idx]], y[order[r_idx:]]])
        validation = X[order[l_idx:r_idx]], y[order[l_idx:r_idx]]
        yield train, validation
        l_idx, r_idx = r_idx, r_idx + fold_width
