################################################################################
# CSE 253: Programming Assignment 2
# Code snippet by Eric Yang Yu, Ajit Kumar, Savyasachi
# Winter 2022
################################################################################

from turtle import shape
import numpy as np
import math


class Activation:
    """
    The class implements different types of activation functions for
    your neural network layers.

    Example (for sigmoid):
        >>> sigmoid_layer = Activation("sigmoid")
        >>> z = sigmoid_layer(a)
        >>> gradient = sigmoid_layer.backward(delta)
    """

    def __init__(self, activation_type="sigmoid"):
        """
        Initialize activation type and placeholders here.
        """
        if activation_type not in ["sigmoid", "tanh", "ReLU"]:
            raise NotImplementedError("%s is not implemented." % (activation_type))

        # Type of non-linear activation.
        self.activation_type = activation_type
        # Placeholder for input. This will be used for computing gradients.
        self.x = None

        # placeholder for output y or z = g(a). Also be used for computing gradients.
        self.output = None

    def __call__(self, a):
        """
        This method allows your instances to be callable.
        """
        return self.forward(a)

    def forward(self, a):
        """
        Compute the forward pass.
        """
        if self.activation_type == "sigmoid":
            return self.sigmoid(a)

        elif self.activation_type == "tanh":
            return self.tanh(a)

        elif self.activation_type == "ReLU":
            return self.ReLU(a)

    def backward(self, delta):
        """
        Compute the backward pass.
        """
        if self.activation_type == "sigmoid":
            grad = self.grad_sigmoid()

        elif self.activation_type == "tanh":
            grad = self.grad_tanh()

        elif self.activation_type == "ReLU":
            grad = self.grad_ReLU()

        return grad * delta

    def sigmoid(self, x):
        """
        Implement the sigmoid activation here.
        """
        self.x = x
        self.output = 1 / (1+np.exp(-x))
        return self.output

    def tanh(self, x):
        """
        Implement tanh here.
        """
        self.x = x
        self.output = np.tanh(x)
        return self.output

    def ReLU(self, x):
        """
        Implement ReLU here.
        """
        self.x = x
        self.output = np.maximum(x, 0)
        return self.output

    def grad_sigmoid(self):
        """
        Compute the gradient for sigmoid here.
        """
        return self.output * (1-self.output)

    def grad_tanh(self):
        """
        Compute the gradient for tanh here.
        """
        return 1 - self.output * self.output

    def grad_ReLU(self):
        """
        Compute the gradient for ReLU here.
        """
        toggle = self.x > 0
        return toggle


class Layer:
    """
    This class implements Fully Connected layers for your neural network.

    Example:
        >>> fully_connected_layer = Layer(1024, 100)
        >>> output = fully_connected_layer(input)
        >>> gradient = fully_connected_layer.backward(delta)
    """

    def __init__(self, in_units, out_units):
        """
        Define the architecture and create placeholder.
        """
        np.random.seed(42)
        self.w = math.sqrt(2 / in_units) * np.random.randn(in_units,
                                                           out_units)  # You can experiment with initialization.
        self.b = np.zeros((1, out_units))  # Create a placeholder for Bias
        self.x = None  # Save the input to forward in this
        self.a = None  # Save the output of forward pass in this (without activation)

        self.d_x = None  # Save the gradient w.r.t x in this
        self.d_w = None  # Save the gradient w.r.t w in this
        self.d_b = None  # Save the gradient w.r.t b in this

        self.delta_history_w = 0 # used for momentum, momentum depends on history gradients
        self.delta_history_b = 0 # used for momentum, momentum depends on history gradients

        self.w_best = self.w # used for select the best model
        self.b_best = self.b # used for select the best model

    def __call__(self, x):
        """
        Make layer callable.
        """
        return self.forward(x)

    def forward(self, x):
        """
        Compute the forward pass through the layer here.
        Do not apply activation here.
        Return self.a
        """
        self.x = x

        # a = x^{T}w + b
        self.a = np.matmul(self.x, self.w) + self.b
        return self.a

    def backward(self, delta, l2_penalty=0, l1_penalty=0):
        """
        Write the code for backward pass. This takes in gradient from its next layer as input,
        computes gradient for its weights and the delta to pass to its previous layers.
        Return self.d_x (recursive usage of deltas)
        """
        N, d = self.x.shape

        # calculate dE/da, and dE/db here, formulae derived from part1(c) of the homework.
        self.d_x = np.matmul(delta, self.w.T)
        self.d_b = np.sum(delta, axis=0) / N

        # calculate dE/dw here, with different regularizations, it will update differently
        if l2_penalty:
            self.d_w = np.matmul(self.x.T, delta) / N - l2_penalty * self.w
        elif l1_penalty:
            self.d_w = np.matmul(self.x.T, delta) / N - l1_penalty * (self.w / np.abs(self.w))
        else:
            self.d_w = np.matmul(self.x.T, delta) / N

        return self.d_x

    def save_weights(self):
        self.w_best = self.w
        self.b_best = self.b

    def load_weights(self):
        self.w = self.w_best
        self.b = self.b_best

    def weights_update(self, lr, momentum=False, gamma=None):
        """
        update the weights with the delta rule. Past deltas will also be stored
        and utilized for computing the momentum.
        """

        # the basic delta computed here
        delta_w = lr * self.d_w
        delta_b = lr * self.d_b

        # if momentum triggered, compute the momentum here
        if momentum:
            delta_w = delta_w + gamma * self.delta_history_w
            delta_b = delta_b + gamma * self.delta_history_b
            self.delta_history_w = delta_w
            self.delta_history_b = delta_b

        # update the weights at the end
        self.w = self.w + delta_w
        self.b = self.b + delta_b

class NeuralNetwork:
    """
    Create a Neural Network specified by the input configuration.

    Example:
        >>> net = NeuralNetwork(config)
        >>> output = net(input)
        >>> net.backward()
    """

    def __init__(self, config):
        """
        Create the Neural Network using config.
        """
        self.layers = []  # Store all layers in this list.
        self.x = None  # Save the input to forward in this
        self.y = None  # Save the output vector of model in this
        self.targets = None  # Save the targets in forward in this variable

        # Add layers specified by layer_specs.
        for i in range(len(config['layer_specs']) - 1):
            self.layers.append(Layer(config['layer_specs'][i], config['layer_specs'][i + 1]))
            if i < len(config['layer_specs']) - 2:
                self.layers.append(Activation(config['activation']))

    def __call__(self, x, targets=None):
        """
        Make NeuralNetwork callable.
        """
        return self.forward(x, targets)

    def save_weights(self):
        for layer in self.layers:
            if isinstance(layer, Layer):
                layer.save_weights()

    def load_weights(self):
        for layer in self.layers:
            if isinstance(layer, Layer):
                layer.load_weights()

    def forward(self, x, targets=None, l2_penalty=0, l1_penalty=0):
        """
        Compute forward pass through all the layers in the network and return it.
        If targets are provided, return loss as well.
        """
        self.x = x
        self.targets = targets

        # Forward pass for inputs
        for layer in self.layers:
            x = layer.forward(x)

        # output for Softmax
        self.y = self.softmax(x)

        # Compute cross-entropy loss
        loss = self.loss(self.y, self.targets)

        # with different ways of regularization, the loss will need a further updation w.r.t complexity.
        if l2_penalty:
            for layer in self.layers:
                if isinstance(layer, Layer):
                    loss += (np.sum(layer.w ** 2)) * l2_penalty / 2
        elif l1_penalty:
            for layer in self.layers:
                if isinstance(layer, Layer):
                    loss += (np.sum(np.abs(layer.w))) * l1_penalty
        return loss

    def backward(self, l2_penalty=0, l1_penalty=0):
        """
        Implement backpropagation here.
        Call backward methods of individual layer's.
        """
        # output layer at the top, d = t - y
        delta = self.targets - self.y

        # go all the way to the bottom layer and do the updates       
        for layer in self.layers[::-1]:

            # if the current layer is a hidden/input layer, perform the matrix product
            if isinstance(layer, Layer):
                delta = layer.backward(delta, l2_penalty, l1_penalty)

            # otherwise, take in the derivative of the activation function
            else:
                delta = layer.backward(delta)

    # update the weights, from top layer to the bottom
    def weights_update(self, lr, momentum=False, gamma=None):
        for layer in self.layers[::-1]:
            if isinstance(layer, Layer):
                layer.weights_update(lr, momentum, gamma)

    def testAccuracy(self, x, targets):
        """
        Testing the output accuracy by comparing with targets
        """
        for layer in self.layers:
            x = layer.forward(x)

        # select the choice of the model from the output
        # get the actual category from targets
        outs = np.argmax(self.softmax(x), axis=1)
        targets = np.argmax(targets, axis=1)

        # get number of accurate predictions, and calculate accuracy
        accurate_samples = np.sum(targets == outs)
        return accurate_samples / len(targets)

    def softmax(self, x):
        """
        Implement the softmax function here.
        Remember to take care of the overflow condition.
        """

        # Overflow is handled here, we sumtract the array's maximum from each entry.
        # Therefore, the magnitude of the exponents will be small.
        shift_exps = np.exp(x - np.max(x, axis=1, keepdims=True))

        # formula for multi-valued softmax
        return shift_exps / np.sum(shift_exps, axis=1, keepdims=True)

    def loss(self, logits, targets):
        """
        Compute the categorical cross-entropy loss and return it.
        """

        # calculate the cross-entropy loss
        N, d = targets.shape
        return - np.sum(targets * np.log(logits)) / N
