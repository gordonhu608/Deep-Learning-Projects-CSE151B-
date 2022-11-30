################################################################################
# CSE 151b: Programming Assignment 2
# Code snippet by Eric Yang Yu, Ajit Kumar, Savyasachi
# Winter 2022
################################################################################
# To install PyYaml, refer to the instructions for your system:
# https://pyyaml.org/wiki/PyYAMLDocumentation
################################################################################
# If you don't have NumPy installed, please use the instructions here:
# https://scipy.org/install.html
################################################################################
import argparse

from data import load_data, load_config, generate_k_fold_set, z_score_normalize, one_hot_encoding, shuffle
from train import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_mlp', dest='train_mlp', action='store_true', default=False,
                        help='Train a single multi-layer perceptron using configs provided in config.yaml')
    parser.add_argument('--check_gradients', dest='check_gradients', action='store_true', default=False,
                        help='Check the network gradients computed by comparing the gradient computed using'
                             'numerical approximation with that computed as in back propagation.')
    parser.add_argument('--regularization', dest='regularization', action='store_true', default=False,
                        help='Experiment with weight decay added to the update rule during training.')
    parser.add_argument('--activation', dest='activation', action='store_true', default=False,
                        help='Experiment with different activation functions for hidden units.')
    parser.add_argument('--topology', dest='topology', action='store_true', default=False,
                        help='Experiment with different network topologies.')
    return parser.parse_args()


if __name__ == "__main__":
    # Parse arguments
    args = parse_args()

    # Load the configuration.
    config = load_config("./config.yaml")

    # Load the data
    train_data, (x_test, y_test) = load_data(), load_data(train=False)

    # Create validation set out of training data.
    X,y = shuffle(train_data)
    split = int(len(X) * 0.8)
    x_train, y_train, x_val, y_val = X[:split], y[:split], X[split:], y[split:]

    # Any pre-processing on the datasets goes here.
    x_train = z_score_normalize(x_train)[0]
    x_val = z_score_normalize(x_val)[0]
    y_train = one_hot_encoding(y_train)
    y_val = one_hot_encoding(y_val)
    x_test = z_score_normalize(x_test)[0]
    y_test = one_hot_encoding(y_test)
    # Run the writeup experiments here
    if args.train_mlp:
        train_mlp(x_train, y_train, x_val, y_val, x_test, y_test, config)
    if args.check_gradients:
        check_gradients(x_train, y_train, config)
    if args.regularization:
        regularization_experiment(x_train, y_train, x_val, y_val, x_test, y_test, config)
    if args.activation:
        activation_experiment(x_train, y_train, x_val, y_val, x_test, y_test, config)
    if args.topology:
        topology_experiment(x_train, y_train, x_val, y_val, x_test, y_test, config)
