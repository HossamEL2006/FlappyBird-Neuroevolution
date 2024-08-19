"""
##############################################################################
#                                                                            #
#                       © 2024 HossamEL2006                                  #
#                                                                            #
#   You are permitted to use, copy, and modify this file for personal,       #
#   non-commercial purposes only. Redistribution of any modified or          #
#   unmodified version of this file, especially for commercial purposes,     #
#   is not allowed without explicit permission from the author.              #
#                                                                            #
##############################################################################

This module implements a simple Neural Network class and a Layer class to represent
a feedforward neural network with customizable architecture and activation functions.

This script was mostly documented and commented by ChatGPT
"""

import numpy as np

ACTIVATIONS = {
    'sigmoid': lambda z: 1 / (1 + np.exp(-z)),
    'relu': lambda z: np.maximum(0, z),
    'binary_step': lambda z: np.where(z >= 0, 1, 0)
}


class NeuralNetwork:
    """A simple implementation of a neural network with customizable architecture and
       activation functions.

    Attributes:
        architecture (list): The architecture of the neural network, where each element represents
                             the number of neurons in that layer.
        activation (function): The activation function used by the neurons in the network.
        layers (list): List of Layer objects representing each layer in the neural network.
        n_inputs (int): Number of input neurons.
        hidden_layers_architecture (list): Architecture of hidden layers.
        n_outputs (int): Number of output neurons.
        n_layers (int): Total number of layers in the network.
        n_neurons (int): Total number of neurons in the network.
        n_weights (int): Total number of weights in the network.
        n_biases (int): Total number of biases in the network.
        n_parametres (int): Total number of parameters (weights + biases) in the network.
    """

    def __init__(self, architecture, activation_function='sigmoid'):
        """
        Initialize the NeuralNetwork with a given architecture and activation function.

        Args:
            architecture (list[int]): List specifying the number of neurons in each layer.
            activation_function (str, optional): Activation function name. Defaults to 'sigmoid'.
        """
        self.architecture = architecture
        self.activation = ACTIVATIONS[activation_function]
        self.initialize_constants()
        self.initialize_layers()

    def initialize_constants(self):
        """
        Initializes the constants that describe the network's architecture.

        This method sets the number of inputs, outputs, layers, neurons, weights, biases, and
        total parameters.
        """
        self.n_inputs = self.architecture[0]
        self.hidden_layers_architecture = self.architecture[1:-1]
        self.n_outputs = self.architecture[-1]
        self.n_layers = len(self.architecture)
        self.n_neurons = sum(self.architecture)
        self.n_weights = sum(
            [a * b for a, b in zip(self.architecture, self.architecture[1:])])
        self.n_biases = sum(self.architecture[1:])
        self.n_parametres = self.n_weights + self.n_biases

    def initialize_layers(self):
        """
        Initializes the layers of the network based on the architecture.

        The first layers is not considered a layer since it's just the input_layer, so no need to
        create a Layer object for it.
        """
        self.layers = []
        for layer_index, (n_neurons, n_inputs) in enumerate(zip(self.architecture[1:],
                                                                self.architecture[:-1]),
                                                            start=1):
            self.layers.append(
                Layer(n_neurons, n_inputs, layer_index, self.activation))

    def forward(self, x):
        """
        Performs forward propagation through the network.

        Args:
            x (np.ndarray): Input data to the network, with shape (n_inputs, n_samples).

        Returns:
            activations (list): The array outputted by the last Layer of the Neural Network.

        Raises:
            ValueError: If the input data does not match the number of input neurons.
        """
        if x.shape[0] != self.n_inputs:
            raise ValueError(
                f"NN takes {self.n_inputs} inputs, but {x.shape[0]} were given")
        for layer in self.layers:
            x = layer.compute_activation(x)
        return x

    def mutate(self, mutation_rate, mutation_value=None):
        """
        Mutates the weights and biases of the network's layers.

        Args:
            mutation_rate (float): Probability of each parameter being mutated.
            mutation_value (float, optional): The range within which the parameter values can be
                                              mutated. If None, random values are used.
                                              Defaults to None.
        """
        for layer in self.layers:
            layer.mutate(mutation_rate, mutation_value)

    def print_info(self):
        """
        Prints detailed information about the network's layers, including weights, biases, and
        other parameters.
        """
        for layer in self.layers:
            print(f'Layer: L{layer.layer_index}')
            print('Weights:')
            print(layer.weights)
            print('Biases:')
            print(layer.biases)
        print(f'Number of layers: {self.n_layers}')
        print(f'Number of neurons: {self.n_neurons}')
        print(f'Number of weights: {self.n_weights}')
        print(f'Number of biases: {self.n_biases}')
        print(f'Number of parameters: {self.n_parametres}')


class Layer:
    """Represents a single layer in a neural network.

    Attributes:
        n_neurons (int): Number of neurons in the layer.
        weights (np.ndarray): Weights matrix of the layer.
        biases (np.ndarray): Biases vector of the layer.
        layer_index (int): The index of the layer in the network.
        activation (function): The activation function used by the layer.
    """

    def __init__(self, n_neurons, n_inputs, layer_index, activation):
        """
        Initialize the Layer with given parameters.

        Args:
            n_neurons (int): Number of neurons in the layer.
            n_inputs (int): Number of inputs to the layer.
            layer_index (int): The index of the layer in the network.
            activation (function): The activation function to be used by the layer.
        """
        self.n_neurons = n_neurons  # Fixed variable name from n_neuron to n_neurons
        self.weights = np.random.normal(size=(n_neurons, n_inputs))
        self.biases = np.random.normal(size=(n_neurons, 1))
        self.layer_index = layer_index
        self.activation = activation

    def compute_activation(self, x):
        """
        Computes the activation for the layer given an input.

        Args:
            x (np.ndarray): Input data to the layer, with shape (n_inputs, n_samples).

        Returns:
            np.ndarray: The output of the layer after applying the weights, biases, and
                        activation function.
        """
        return self.activation(np.dot(self.weights, x) + self.biases)

    def mutate(self, mutation_rate, mutation_value=None):
        """
        Mutates the layer's weights and biases.

        Args:
            mutation_rate (float): Probability of each parameter being mutated.
            mutation_value (float, optional): The range within which the parameter values can be
                                              mutated. If None, random values are used.
                                              Defaults to None.
        """
        self.weights = mutate_array(
            self.weights, mutation_rate, mutation_value)
        self.biases = mutate_array(self.biases, mutation_rate, mutation_value)


def mutate_value(value, probability, mutation_value=None):
    """
    Mutates a single value based on a given probability and mutation range.

    Args:
        value (float): The original value to be mutated.
        probability (float): Probability of the value being mutated.
        mutation_value (float, optional): The range within which the value can be mutated.
                                          If None, a random normal value is returned.
                                          Defaults to None.

    Returns:
        float: The mutated value.
    """
    if np.random.rand() < probability:
        if mutation_value is None:
            return np.random.normal()
        else:
            return value + np.random.uniform(-mutation_value, mutation_value)
    else:
        return value


mutate_array = np.vectorize(mutate_value)
""" mutate_array is a function that takes as input a numpy ndarray and runs the function
    mutate_value over each number of the matrix while replacing them by their mutated value.
"""
