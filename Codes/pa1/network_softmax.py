import numpy as np
import data_softmax
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
    return np.exp(a) / np.sum(np.exp(a))

class Binary_cross_entropy:
    def __init__(self):
        pass
    def __call__(self, y, t):
        return t * np.log(y) + (1 - t) * np.log(1 - y)
    def backword(y, t, X):
        return np.matmul(X.T, (t-y))
        

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
    return -np.sum(t * np.log(y))

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
        self.weights = np.zeros((28*28+1, 10))

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
        X = data_softmax.z_score_normalize(X)
        y = data_softmax.onehot_encode(y).reshape(-1, 10)
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
        temp_weight = self.weights = np.zeros((28*28+1, 10))
        tempAcc = 0.0
        for fold in range(1, self.k_folds+1):
            val_X, val_y = X_folds[fold - 1], y_folds[fold - 1]
            train_X, train_y = np.delete(X_folds, obj=fold-1, axis=0), np.delete(y_folds, obj=fold-1, axis=0)
            train_X = train_X.reshape(-1, train_X.shape[2])
            train_y = train_y.reshape(-1, train_y.shape[2])

            self.weights = np.zeros((28*28+1, 10))
            for e in range(self.epochs): #self.epochs
                running_loss = 0.0
                running_corrects = 0.0
                val_running_loss = 0.0
                val_running_corrects = 0.0
                sum_deriv = np.zeros((28*28+1, 10))

                indices = np.arange(train_X.shape[0])
                np.random.shuffle(indices)
                train_X, train_y = train_X[indices], train_y[indices]

                for i in range(0, train_X.shape[0]):
                    
                    inputs, labels = train_X[i:i+1], train_y[i:i+1]
                    inputs = np.insert(inputs, 0 , 1, axis=1) #vector
                    outputs = self.forward(inputs) #vector 1x10
                    
                    loss = self.loss(outputs, labels) #singlenumber
                    running_loss += loss

                    outputs_arr = outputs[0]
                    labels_arr = labels[0]
                    if labels_arr[np.argmax(outputs_arr)] == 1:
                        running_corrects += 1
                       
                    # backward
                    for t,y in zip(labels,outputs):
                        t = t.reshape(1,-1)
                        y = y.reshape(1,-1)

                        sum_deriv = np.add(sum_deriv, Binary_cross_entropy.backword(y,t, inputs))


                    if (i % self.batch_size == 0):
                        self.weights = self.weights + self.lr * sum_deriv / self.batch_size
                        sum_deriv = np.zeros((28*28+1, self.out_dim))
            
                #testing for the reamining batch
                for val_inputs, val_labels in zip(val_X, val_y):
                    
                    val_inputs = np.insert(val_inputs, 0 , 1) #vector
                    val_outputs = self.forward(val_inputs) #number
                    
                    #print(val_labels, val_outputs)
                    if val_labels[np.argmax(val_outputs)] == 1:
                        val_running_corrects += 1
                    
                    val_outputs = val_outputs.reshape(1,-1)
                    val_labels = val_labels.reshape(1,-1)
                    #print(val_outputs.shape, val_labels.shape)

                    val_loss = self.loss(val_outputs, val_labels) #number
                    #print("val_loss", val_loss)
                    val_running_loss += val_loss

                epoch_loss = running_loss/len(train_X) / 10
                epoch_acc = running_corrects/len(train_X)
                running_loss_history.append(epoch_loss)
                running_corrects_history.append(epoch_acc)

                val_epoch_loss = val_running_loss/len(val_X) / 10
                val_epoch_acc = val_running_corrects / len(val_X)
                val_running_loss_history.append(val_epoch_loss)
                val_running_corrects_history.append(val_epoch_acc)

                print('epoch :', (e+1))
                print('training loss: {:.4f}, acc {:.4f} '.format(epoch_loss, epoch_acc))
                print('validation loss: {:.4f}, validation acc {:.4f} '.format(val_epoch_loss, val_epoch_acc))

            #plot_stats(running_corrects_history, running_loss_history, val_running_corrects_history, val_running_loss_history)
            cumulativeAcc += val_epoch_acc
            if tempAcc < val_epoch_acc:
                temp_weight = self.weights

        self.weights = temp_weight

        # for hyperparameter tuning
        cumulativeAcc = cumulativeAcc / self.k_folds
        print("cumulativeacc", cumulativeAcc)
            
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
        X = data_softmax.z_score_normalize(X)
        y = data_softmax.onehot_encode(y).reshape(-1, 10)

        test_running_corrects = 0.0
        test_running_loss = 0.0

        for val_inputs, val_labels in zip(X, y):
            
            val_inputs = np.insert(val_inputs, 0 , 1)
            val_outputs = self.forward(val_inputs)
            
            if val_labels[np.argmax(val_outputs)] == 1:
                test_running_corrects += 1
                

            val_outputs = val_outputs.reshape(1,-1)
            val_labels = val_labels.reshape(1,-1)
            val_loss = self.loss(val_outputs, val_labels) 
            
            test_running_loss += val_loss

        test_running_loss = test_running_loss / len(X) / 10
        test_running_acc = test_running_corrects / len(X)

        print("result:", test_running_loss, test_running_acc)
        return test_running_loss, test_running_acc
        
