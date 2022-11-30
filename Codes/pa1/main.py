import argparse
import network
from network import Network
import data
import image

def main(hyperparameters):
    
    loss = network.binary_cross_entropy
    act = network.sigmoid
    net = Network(hyperparameters, act, loss, 1)
    train_data = data.load_data()
    net.train(train_data)
    test_data = data.load_data(False)
    net.test(test_data)
    
    
parser = argparse.ArgumentParser(description = 'CSE151B PA1')
parser.add_argument('--batch-size', type = int, default = 128,
        help = 'input batch size for training (default: 1)')
parser.add_argument('--epochs', type = int, default = 100,
        help = 'number of epochs to train (default: 100)')
parser.add_argument('--learning-rate', type = float, default = 0.01,
        help = 'learning rate (default: 0.001)', dest='lr')
parser.add_argument('--z_score', dest = 'normalization', action='store_const',
        default = data.min_max_normalize, const = data.z_score_normalize,
        help = 'use z-score normalization on the dataset, default is min-max normalization')
parser.add_argument('--k-folds', type = int, default = 10,
        help = 'number of folds for cross-validation')
parser.add_argument('--patience', type = int, default = 100,
        help = 'How long to wait after last time validation loss improved')

hyperparameters = parser.parse_args()
main(hyperparameters)
