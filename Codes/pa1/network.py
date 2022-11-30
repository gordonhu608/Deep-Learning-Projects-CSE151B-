import numpy as np
import data
import time
import matplotlib as plt
#import tqdm

"""
NOTE
----
Start by implementing your methods in non-vectorized format - use loops and other basic programming constructs.
Once you're sure everything works, use NumPy's vector operations (dot products, etc.) to speed up your network.
"""

def sigmoid(a):
    """
    Compute the sigmoid function.

    f(x) = 1 / (1 + e ^ (-x))

    Parameters
    ----------
    a
        The internal value while a pattern goes through the network
    Returns
    -------
    float
       Value after applying sigmoid (z from the slides).
    """
    return 1 / (1 + np.exp(-a))

def softmax(a):
    """
    Compute the softmax function.

    f(x) = (e^x) / Σ (e^x)

    Parameters
    ----------
    a
        The internal value while a pattern goes through the network
    Returns
    -------
    float
       Value after applying softmax (z from the slides).
    """
    return (np.exp(a)) / sum(np.exp(a))

class Binary_cross_entropy:
    def __init__(self):
        pass
    def __call__(self, y, t):
        return t * np.log(y) + (1 - t) * np.log(1 - y)
    def backword(y, t, X):
        #print(X.shape)
        #print(t.shape)
        return X.T*(t-y)
        

def binary_cross_entropy(y, t):
    """
    Compute binary cross entropy.

    L(x) = t*ln(y) + (1-t)*ln(1-y)

    Parameters
    ----------
    y
        The network's predictions
    t
        The corresponding targets
    Returns
    -------
    float 
        binary cross entropy loss value according to above definition
    """
    return -(t * np.log(y) + (1 - t) * np.log(1 - y))

def multiclass_cross_entropy(y, t):
    """
    Compute multiclass cross entropy.

    L(x) = - Σ (t*ln(y))

    Parameters
    ----------
    y
        The network's predictions
    t
        The corresponding targets
    Returns
    -------
    float 
        multiclass cross entropy loss value according to above definition
    """
    return -sum(t * np.log(y))

class Network:
    def __init__(self, hyperparameters, activation, loss, out_dim):
        """
        Perform required setup for the network.

        Initialize the weight matrix, set the activation function, save hyperparameters.

        You may want to create arrays to save the loss values during training.

        Parameters
        ----------
        hyperparameters
            A Namespace object from `argparse` containing the hyperparameters
        activation
            The non-linear activation function to use for the network
        loss
            The loss function to use while training and testing
        """
        self.hyperparameters = hyperparameters
        self.activation = activation
        self.loss = loss
        self.out_dim = out_dim 
        self.weights = np.zeros((28*28+1, self.out_dim))

        # self-added decomposed parameters
        hyper_dict = vars(hyperparameters)
        self.batch_size = hyper_dict['batch_size']
        self.epochs = hyper_dict['epochs']
        self.lr = hyper_dict['lr']
        self.norm = hyper_dict['normalization']
        self.k_folds = hyper_dict['k_folds']
        self.patience = hyper_dict['patience']

    def forward(self, X):
        """
        Apply the model to the given patterns

        Use `self.weights` and `self.activation` to compute the network's output

        f(x) = σ(w*x)
            where
                σ = non-linear activation function
                w = weight matrix

        Make sure you are using matrix multiplication when you vectorize your code!

        Parameters
        ----------
        X
            Patterns to create outputs for
        """
        return self.activation(np.matmul(X, self.weights))

    def __call__(self, X):
        return self.forward(X)
    def plot_stats(train_acc, train_loss, val_acc, val_loss):
        """
        Plot training stats
        """
        f = plt.figure(figsize=(12,5))
        ax = f.add_subplot(121)
        ax2 = f.add_subplot(122)
        # plot training and validation accuracy
        ax.plot(train_acc, label = "Training accuracy")
        ax.plot(val_acc, label = "Validation accuracy")
        ax.legend(loc='upper left')
        # plot training and validation loss
        ax2.plot(train_loss, label = "Training loss")
        ax2.plot(val_loss, label = "Validation loss")
        ax2.legend(loc='upper right')
        plt.show()

    def train(self, minibatch):
        """
        Train the network on the given minibatch

        Use `self.weights` and `self.activation` to compute the network's output
        Use `self.loss` and the gradient defined in the slides to update the network.

        Parameters
        ----------
        minibatch
            The minibatch to iterate over

        Returns
        -------
        tuple containing:
            average loss over minibatch
            accuracy over minibatch
        """
        X, y = minibatch
        X = data.min_max_normalize(X)

        boolArr = (y == 2) | (y == 6)
        X = X[boolArr]
        y = y[boolArr]
        y = (y-2) / 4
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        X, y = X[indices], y[indices]

        X_folds = np.array_split(X, self.k_folds)
        y_folds = np.array_split(y, self.k_folds)
        
        running_loss_history = []
        running_corrects_history = []
        val_running_loss_history = []
        val_running_corrects_history = []
        cumulativeAcc = 0.0

        for fold in range(1, self.k_folds + 1): 
            val_X, val_y = X_folds[fold - 1], y_folds[fold - 1]
            train_X, train_y = np.delete(X_folds, obj=fold-1, axis=0), np.delete(y_folds, obj=fold-1, axis=0)
            train_X = train_X.reshape(-1, train_X.shape[2])
            train_y = train_y.reshape(-1)
            #print(train_y.shape)
            #print(train_X.shape)
            self.weights = np.zeros((28*28+1, self.out_dim))
            early_stop_acc = []

            for e in range(self.epochs):
                running_loss = 0.0
                running_corrects = 0.0
                val_running_loss = 0.0
                val_running_corrects = 0.0
                
                indices = np.arange(train_X.shape[0])
                np.random.shuffle(indices)
                train_X, train_y = train_X[indices], train_y[indices]
                sum_deriv = np.zeros((28*28+1, self.out_dim))
                
                for i in range(0, train_X.shape[0]):
                    #start = i
                    #end = i + self.batch_size
                    #if end >= train_X.shape[0]:
                    #    continue
                    
                    inputs, labels = train_X[i:i+1], train_y[i:i+1]
                    inputs = np.insert(inputs, 0 , 1, axis=1) #vector
                    #print(inputs.shape)
                    outputs = self.forward(inputs) #singlenumber
                    #print(outputs.shape)
                    labels = labels.reshape(1,-1)
                    #print(labels.shape)
                    loss = self.loss(outputs, labels) #singlenumber
                    running_loss += loss
                    if outputs-labels < 0.5 and outputs-labels > -0.5:
                        running_corrects += 1
                    
                    # backward
                    for t,y in zip(labels,outputs):
                        #print(t,y)
                        #print(Binary_cross_entropy.backword(y,t, inputs).shape)
                        sum_deriv += Binary_cross_entropy.backword(y,t, inputs)


                    if (i % self.batch_size == 0):
                        self.weights = self.weights + self.lr * sum_deriv / self.batch_size
                        sum_deriv = np.zeros((28*28+1, self.out_dim))
                        #print("weights:", self.weights)
            
                #testing for the reamining batch
                for val_inputs, val_labels in zip(val_X, val_y):
                    
                    val_inputs = np.insert(val_inputs, 0 , 1) #vector
                    val_outputs = self.forward(val_inputs) #number
                    #print(val_outputs, val_labels)
                    val_loss = self.loss(val_outputs, val_labels) #number
                    #print("val_loss", val_loss)
                    val_running_loss += val_loss
                    if val_outputs-val_labels < 0.5 and val_outputs-val_labels > -0.5:
                        val_running_corrects += 1

                epoch_loss = running_loss/len(train_X)
                epoch_acc = running_corrects/len(train_X)
                running_loss_history.append(epoch_loss)
                running_corrects_history.append(epoch_acc)

                val_epoch_loss = val_running_loss/len(val_X)
                val_epoch_acc = val_running_corrects / len(val_X)
                val_running_loss_history.append(val_epoch_loss)
                val_running_corrects_history.append(val_epoch_acc)

                early_stop_acc.append(val_epoch_acc)
                if len(early_stop_acc) > self.patience:
                    last = early_stop_acc[::-1][:self.patience]
                    best = early_stop_acc[self.patience-1]
                    if np.all(last < best):
                        break

                print('epoch :', (e+1))
                print('training loss: {:.4f}, acc {:.4f} '.format(epoch_loss[0][0], epoch_acc))
                print('validation loss: {:.4f}, validation acc {:.4f} '.format(val_epoch_loss[0], val_epoch_acc))
                #plot_stats(running_corrects_history, running_loss_history, val_running_corrects_history, val_running_loss_history)

            # for hyperparameter tuning              
            cumulativeAcc += val_epoch_acc

        cumulativeAcc /= self.k_folds
        print("cumulativeAcc:", cumulativeAcc)
            
    def test(self, minibatch):
        """
        Test the network on the given minibatch

        Use `self.weights` and `self.activation` to compute the network's output
        Use `self.loss` to compute the loss.
        Do NOT update the weights in this method!

        Parameters
        ----------
        minibatch
            The minibatch to iterate over

        Returns
        -------
            tuple containing:
                average loss over minibatch
                accuracy over minibatch
        """
        X, y = minibatch
        X = data.min_max_normalize(X)

        boolArr = (y == 2) | (y == 6)
        X = X[boolArr]
        y = y[boolArr]
        y = (y-2) / 4
        test_running_corrects = 0.0
        test_running_loss = 0.0

        for val_inputs, val_labels in zip(X, y):
            
            val_inputs = np.insert(val_inputs, 0 , 1) #vector
            val_outputs = self.forward(val_inputs) #number
            
            val_loss = self.loss(val_outputs, val_labels) #number
            test_running_loss += val_loss

            if val_outputs-val_labels < 0.5 and val_outputs-val_labels > -0.5:
                test_running_corrects += 1

        test_running_loss = test_running_loss / len(X)
        test_running_acc = test_running_corrects / len(X)

        print("result:", test_running_loss, test_running_acc)
        return test_running_loss, test_running_acc
        
