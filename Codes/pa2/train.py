################################################################################
# CSE 151b: Programming Assignment 2
# Code snippet by Eric Yang Yu, Ajit Kumar, Savyasachi
# Winter 2022
################################################################################
from data import write_to_file
from neuralnet import *
#from tqdm import tqdm
import matplotlib.pyplot as plt
import math

# generate minibatches to perform SGD on them
def minibatch(inputs, targets, batch_size):
    
    # shuffle the dataset first
    indices = np.arange(len(inputs))
    np.random.shuffle(indices)
    shuffled_X, shuffled_y = inputs[indices], targets[indices]

    # generate minibatches by selecting segments from the shuffled dataset
    for start in range(0, len(shuffled_X) - batch_size + 1, batch_size):
        yield shuffled_X[start: start + batch_size], shuffled_y[start: start + batch_size]


def train(x_train, y_train, x_val, y_val, config):
    """
    Train your model here using batch stochastic gradient descent and early stopping. Use config to set parameters
    for training like learning rate, momentum, etc.

    Args:
        x_train: The train patterns
        y_train: The train labels
        x_val: The validation set patterns
        y_val: The validation set labels
        config: The configs as specified in config.yaml
        experiment: An optional dict parameter for you to specify which experiment you want to run in train.

    Returns:
        5 things:
            training and validation loss and accuracies - 1D arrays of loss and accuracy values per epoch.
            best model - an instance of class NeuralNetwork. You can use copy.deepcopy(model) to save the best model.
    """
    epochs = config['epochs']
    batch_size = config['batch_size']
    momentum = config['momentum']
    gamma = config['momentum_gamma']
    lr = config['learning_rate']
    early_stop = config['early_stop']
    early_stop_epoch = config['early_stop_epoch']
    l2_penalty = config['L2_penalty']
    l1_penalty = config['L1_penalty']

    train_acces = []
    val_acces = []
    train_losses = []
    val_losses = []

    # initialize variables for early stopping
    min_loss = 1000.0
    rising_streak = 0

    model = NeuralNetwork(config=config)

    # based on minibatches, SGD is utilized to update the weights for every epoch
    for epoch in range(epochs):
        train_epoch_loss, train_epoch_acc = [], []
        for x, targets in minibatch(x_train, y_train, batch_size=batch_size):
            train_epoch_loss.append(model.forward(x, targets=targets, l2_penalty=l2_penalty, l1_penalty=l1_penalty))
            model.backward(l2_penalty=l2_penalty, l1_penalty=l1_penalty)
            model.weights_update(lr=lr, momentum=momentum, gamma=gamma)
            train_epoch_acc.append(model.testAccuracy(x, targets=targets))

        # training loss and training accuracy
        train_loss = np.mean(np.array(train_epoch_loss))
        train_acc = np.mean(np.array(train_epoch_acc))

        # validation loss and validation accuracy
        val_loss = model.forward(x_val, targets=y_val)
        val_acc = model.testAccuracy(x_val, targets=y_val)

        # print epoch logs
        print("Epoch: {}, Train_loss: {:.4f}, Train_acc: {:.4f},Val_loss: {:.4f}, Val_acc: {:.4f}"\
            .format(epoch + 1, train_loss, train_acc, val_loss, val_acc))

        train_acces.append(train_acc)
        val_acces.append(val_acc)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Check validation loss, if it increases, set up the rising streak for early stop
        if val_loss < min_loss:
            model.save_weights()
            min_loss = val_loss
            rising_streak = 0
        else:
            rising_streak += 1

        # early stop: terminate the loop to prevent overfitting
        if early_stop:
            if rising_streak > early_stop_epoch:
                break

    model.load_weights()
    best_model = model
    return train_acces, val_acces, train_losses, val_losses, best_model


def test(model, x_test, y_test):
    """
    Does a forward pass on the model and returns loss and accuracy on the test set.

    Args:
        model: The trained model to run a forward pass on.
        x_test: The test patterns.
        y_test: The test labels.

    Returns:
        Loss, Test accuracy
    """
    test_loss = model.forward(x_test, targets=y_test)
    return test_loss, model.testAccuracy(x_test, y_test)


def train_mlp(x_train, y_train, x_val, y_val, x_test, y_test, config):
    """
    This function trains a single multi-layer perceptron and plots its performances.

    NOTE: For this function and any of the experiments, feel free to come up with your own ways of saving data
            (i.e. plots, performances, etc.). A recommendation is to save this function's data and each experiment's
            data into separate folders, but this part is up to you.
    """
    # train the model
    train_acc, valid_acc, train_loss, valid_loss, best_model = \
        train(x_train, y_train, x_val, y_val, config)

    test_loss, test_acc = test(best_model, x_test, y_test)

    print("Config: %r" % config)
    print("Test Loss", test_loss)
    print("Test Accuracy", test_acc)
    f = plt.figure(figsize=(12, 5))
    ax = f.add_subplot(121)
    ax2 = f.add_subplot(122)
    # plot training and validation accuracy
    ax.plot(train_acc, label="Training accuracy")
    ax.plot(valid_acc, label="Validation accuracy")
    ax.legend(loc='upper left')
    # plot training and validation loss
    ax2.plot(train_loss, label="Training loss")
    ax2.plot(valid_loss, label="Validation loss")
    ax2.legend(loc='upper right')
    plt.show()
    # DO NOT modify the code below.
    data = {'train_loss': train_loss, 'val_loss': valid_loss, 'train_acc': train_acc, 'val_acc': valid_acc,
            'best_model': best_model, 'test_loss': test_loss, 'test_acc': test_acc}

    write_to_file('./results.pkl', data)


def activation_experiment(x_train, y_train, x_val, y_val, x_test, y_test, config):
    """
    This function tests all the different activation functions available and then plots their performances.
    """
    for activation in ["sigmoid", "tanh", "ReLU"]:
        config['activation'] = activation
        train_mlp(x_train, y_train, x_val, y_val, x_test, y_test, config)

def topology_experiment(x_train, y_train, x_val, y_val, x_test, y_test, config):
    """
    This function tests performance of various network topologies, i.e. making
    the graph narrower and wider by halving and doubling the number of hidden units.

    Then, we change number of hidden layers to 2 of equal size instead of 1, and keep
    number of parameters roughly equal to the number of parameters of the best performing
    model previously.
    """
    nn_layers = [[784, 64, 10],
                 [784, 256, 10],
                 [784, 128, 128, 10]]
    for layer_specs in nn_layers:
        config['layer_specs'] = layer_specs
        train_mlp(x_train, y_train, x_val, y_val, x_test, y_test, config)


def regularization_experiment(x_train, y_train, x_val, y_val, x_test, y_test, config):
    """
    This function tests the neural network with regularization.
    """
    for weight_decay in [1e-3, 1e-4, 1e-5, 1e-6]:
        config['L2_penalty'] = weight_decay
        train_mlp(x_train, y_train, x_val, y_val, x_test, y_test, config)

    print("L1 Regularization")
    config['L2_penalty'] = 0
    for weight_decay in [1e-3, 1e-6]:
        config['L1_penalty'] = weight_decay
        train_mlp(x_train, y_train, x_val, y_val, x_test, y_test, config)

def check_gradients(x_train, y_train, config):
    """
    Check the network gradients computed by back propagation by comparing with the gradients computed using numerical
    approximation.
    """
    x = x_train
    targets = y_train
    e = 1e-2

    def gradient_bias_check(e, output_idx):
        layer.b[0][output_idx] += e
        loss_add_e = model.forward(x, targets=targets)
        layer.b[0][output_idx] -= 2 * e
        loss_sub_e = model.forward(x, targets=targets)
        layer.b[0][output_idx] += e

        approx = (loss_add_e - loss_sub_e) / (2 * e)
        return approx

    def gradient_weight_check(e, input_idx, output_idx):
        layer.w[input_idx][output_idx] += e
        loss_add_e = model.forward(x, targets=targets)
        layer.w[input_idx][output_idx] -= 2 * e
        loss_sub_e = model.forward(x, targets=targets)
        layer.w[input_idx][output_idx] += e
        
        approx = (loss_add_e - loss_sub_e) / (2 * e)
        return approx

    model = NeuralNetwork(config)

    model.forward(x, targets=targets)
    model.backward()

    def check_within_bound(e1, e2, e):
        if (e1-e2)/ e1 <= e**2:
            print("backprop check pass")

    layernum = 0

    for layer in model.layers:
        if isinstance(layer, Layer):

            # calculating the term (d(W+e) + d(W-e)) / 2e for biases and weights
            d_bias = gradient_bias_check(e, 0)
            d_weight1 = gradient_weight_check(e, 0, 0)
            d_weight2 = gradient_weight_check(e, 5, 5)

            if (layernum < 1):
                print('hidden bias gradient: ', -layer.d_b[0], 'numerical approximation: ', d_bias)
                check_within_bound(d_bias, layer.d_b[0],e)
                print('hidden weight1 gradient: ', -layer.d_w[0][0], 'numerical approximation: ', d_weight1)
                check_within_bound(d_weight1, layer.d_w[0][0], e)
                print('hidden weight2 gradient: ', -layer.d_w[5][5], 'numerical approximation: ', d_weight2)
                check_within_bound(d_weight2, layer.d_w[5][5], e)
            else:
                print('output bias gradient: ', -layer.d_b[0], 'numerical approximation: ', d_bias)
                check_within_bound(d_bias, layer.d_b[0],e)
                print('output weight1 gradient: ', -layer.d_w[0][0], 'numerical approximation: ', d_weight1)
                check_within_bound(d_weight1, layer.d_w[0][0], e)
                print('output weight2 gradient: ', -layer.d_w[5][5], 'numerical approximation: ', d_weight2)
                check_within_bound(d_weight2, layer.d_w[5][5], e)

            layernum += 1

