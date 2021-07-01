#!/usr/bin/envx python3
# -*- coding: utf-8 -*-
""" Created on Fri Nov  8 17:55:27 2019
@author: geem
Constants:
    INSIZE (int): input layer size, number of nodes at input layer
    OUTSIZE (int): output layer size, number of nodes at output layer
    MAXSIZE (int): maximum layer size, maximum number of nodes at any layer
    WPERNODE (int): number of weights per node
    LRNRATE (float): learning rate, or rate of weighting change
    NLAYERS (int): number of layers in the architecture
    NBATCH (int): number of trainig samples in a batch
    DOMAIN (float): domain of variables x, y in bivariate function
        z = f(x,y)
    BA (float): distance between two samples in the domain
"""
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Network hyperparameters
INSIZE = 2
OUTSIZE = 1
MAXSIZE = 8
WPERNODE = 4
LRNRATE = 0.002
NLAYERS = 6
NBATCH = 100
DOMAIN = 20
BA = 2

def f(x, y):
    """Return the image z of a bivariate real function z = f(x,y).
    Arguments:
        x -- GEEM variable x
        y -- GEEM variable y
    """

    #z = ( x**2 + y**2 ) / 100.0     # Paraboloid
    #return z

    return x*y / 50    # Hyperbolic
    #    Hill function
    # return 1.0 - 1.0/(1.0 + (0.5*x)** 1.5) - 1.0/(1.0 + (0.2*y) ** 2.0) + 1.0/(1.0 + (0.5*x) ** 1.5 + (0.2*y) ** 2.0)

    # return 800*(1 - np.exp(-(x*x + y*y)/10000))     # Gaussian

def clip(x):
    """Docstring required"""
    if np.absolute(x) > 1:
        return 1
    
    return np.absolute(x)

def bound(x):
    """Prevents exploding of a given float x by maintaing it within a bound"""
    if x == 0:
        return 0
    if np.log(np.absolute(x)) > 40:
        return np.exp(40) * np.sign(x)
    return x

def activation(w, der, rel_err):
    """Update rule for each node weight in the Protein Network
    w(t+1) = w(t) - activation(w, der, rel_err)

    Args:
        w (float): current weight value
        der (float):
        rel_err (float): relative error
    """
    return LRNRATE \
            * (np.absolute(w) + LRNRATE) \
            * np.sign(der) * np.log(np.absolute(der) + 1) \
            * rel_err

def fplotprep(x_dom, y_dom, Z):
    """Plots a 3D contour for the original bivariate function z=f(x,y)

    Args:
        x_dom (Numpy float array): domain of x
        y_dom (Numpy float array): domain of y
        z (Numpy float array): image of cartesian product x-domain X y_dom
            according to the original function f.
    """
    X, Y = np.meshgrid(x_dom, y_dom)
    Z = f(X, Y)
    fig = plt.figure(1)
    ax = fig.gca(projection = '3d')
    # ax = plt.axes(projection = '3d')
    ax.scatter(X, Y, Z, c = 'r')
    plt.show()

def nplotprep(x_dom, y_dom, z2, layers, weights, layer_sizes):
    """Plots a bivariate function z=F(x,y), given a Protein Network

    A previously-trained Protein Network contains the coefficients of a
    multivariable polynomial regression z = F(x,y) that best approximates
    a given function z = f(x,y)

    Args:
        x_dom (Numpy float array): domain of x
        y_dom (Numpy float array): domain of y
        z2 (Numpy float array): image of cartesian product x-domain X y_dom
            according to the learned function z = F(x, y)

    Vars:
        crnt_lyr_size
        prv_lyr_size

    """
    X, Y = np.meshgrid(x_dom, y_dom)
    z2 = np.zeros((x_dom.size, y_dom.size)).astype(np.float64)

    # Forward pass
    # For each point (x,y) in the meshgrid domain, the trained Protein Network
    # returns an output that corresponds with f(x,y)
    rows = np.size(X, 0)
    columns = np.size(X, 1)
    for row in range(rows):
        for column in range(columns):
            #
            layers[0] = X[row, column]
            layers[1] = Y[row, column]
            current_layer_size = layer_sizes[0]
            nodes_before_current_layer = 0
            for layer in range(1, NLAYERS):
                previous_layer_size = current_layer_size
                current_layer_size = layer_sizes[layer]
                nodes_before_previous_layer = nodes_before_current_layer
                nodes_before_current_layer += previous_layer_size
                for node in range(current_layer_size):

                    # Connectivity between nodes from adjacent layers
                    u1 = 2*node
                    while u1 >= previous_layer_size:
                        u1 -= previous_layer_size
                    u2 = u1 + 1
                    while u2 >= previous_layer_size:
                        u2 -= previous_layer_size

#                     u1 = node
#                     u2 = node + 1
#                     while (u2 >= previous_layer_size):
#                          u2 -= previous_layer_size


                    x1 = bound(layers[nodes_before_previous_layer + u1])
                    x2 = bound(layers[nodes_before_previous_layer + u2])

                    layers[nodes_before_current_layer + node] = \
                            weights[WPERNODE*(nodes_before_current_layer + node)] \
                            + weights[WPERNODE*(nodes_before_current_layer + node) + 1] * x1 \
                            + weights[WPERNODE*(nodes_before_current_layer + node) + 2] * x2 \
                            + weights[WPERNODE*(nodes_before_current_layer + node) + 3] * x1 * x2


            z2[row, column] = layers[nodes_before_current_layer]

    fig = plt.figure(2)
    ax = fig.gca(projection = '3d')
    ax.scatter(X, Y, z2, c = 'b')
    plt.show()


def regression(layers, weights, d_layers, d_weights, layer_sizes):
    """Multivariable polynomial regression by training a Protein Network

    Args:
        layers (Numpy float array): Contains the output of each node.
        weights (Numpy float array): Contains the weights for each node.
        d_layers (Numpy float array): Contains the cumulative gradients,
            relative to the layer outputs of each node.
        d_weights (Numpy float array): Contains the cumulative gradients,
            relative to each weight of each node.
        layer_sizes (Numpy int array): Contains the number of
            nodes at each layer.
    """
    # Seed for random numbers seems to be needed here
    # seed = int(time.monotonic())
    # random.seed(seed)

    rel_err = 0

    for batch in range(NBATCH):
        current_layer_size = layer_sizes[0]
        nodes_before_current_layer = 0
        #seed = int(time.monotonic())
        #random.seed(seed)
        # Forward pass
        # layers[0] = np.random.uniform(low = 0, high = +DOMAIN)
        # layers[1] = np.random.uniform(low = 0, high = +DOMAIN)
        layers[0] = np.random.uniform(low = -10*DOMAIN, high = +10*DOMAIN)
        layers[1] = np.random.uniform(low = -10*DOMAIN, high = +10*DOMAIN)
        for layer in range(1, NLAYERS):
            previous_layer_size = current_layer_size
            current_layer_size = layer_sizes[layer]
            nodes_before_previous_layer = nodes_before_current_layer
            nodes_before_current_layer += previous_layer_size

            for node in range(current_layer_size):
                # Connectivity between nodes from adjacent layers
# =============================================================================
                u1 = 2*node
                while u1 >= previous_layer_size:
                    u1 -= previous_layer_size
                u2 = u1 + 1
                while u2 >= previous_layer_size:
                    u2 -= previous_layer_size
#                 u1 = node
#                 u2 = node + 1
#                 while (u2 >= previous_layer_size):
#                      u2 -= previous_layer_size

                x1 = bound(layers[nodes_before_previous_layer + u1])
                x2 = bound(layers[nodes_before_previous_layer + u2])

                layers[nodes_before_current_layer + node] = \
                        weights[WPERNODE*(nodes_before_current_layer + node)]  \
                        + weights[WPERNODE*(nodes_before_current_layer + node) + 1] * x1  \
                        + weights[WPERNODE*(nodes_before_current_layer + node) + 2] * x2  \
                        + weights[WPERNODE*(nodes_before_current_layer + node) + 3] * x1 * x2

        # Backwards pass
        d_layers[nodes_before_current_layer + layer_sizes[NLAYERS - 1] - 1] = \
            layers[nodes_before_current_layer + layer_sizes[NLAYERS - 1] - 1] \
            - f(layers[0], layers[1])

        rel_err += \
            clip(
                d_layers[nodes_before_current_layer + layer_sizes[NLAYERS - 1] - 1] \
                / f(layers[0], layers[1]))

        previous_layer_size = layer_sizes[NLAYERS - 1]
        nodes_before_previous_layer = nodes_before_current_layer

        #for layer in reversed(range(1,NLAYERS)):
        for layer in range(NLAYERS-1, 0, -1):
            current_layer_size = previous_layer_size
            previous_layer_size = layer_sizes[layer - 1]
            nodes_before_previous_layer -= layer_sizes[layer - 1]


            for node in range(current_layer_size):
                # Connectivity between nodes from adjacent layers
                u1 = 2*node
                while u1 >= previous_layer_size:
                    u1 -= previous_layer_size
                u2 = u1 + 1
                while u2 >= previous_layer_size:
                    u2 -= previous_layer_size
#                u1 = node
#                u2 = node + 1
#                while (u2 >= previous_layer_size):
#                     u2 -= previous_layer_size
                x1 = bound(layers[nodes_before_previous_layer + u1])
                x2 = bound(layers[nodes_before_previous_layer + u2])

                # Derivatives of loss function relative to the weights of current node
                dy = bound(d_layers[nodes_before_current_layer + node])
                d_weights[WPERNODE*(nodes_before_current_layer + node)] += dy
                d_weights[WPERNODE*(nodes_before_current_layer + node) + 1] += dy * x1
                d_weights[WPERNODE*(nodes_before_current_layer + node) + 2] += dy * x2
                d_weights[WPERNODE*(nodes_before_current_layer + node) + 3] += dy * x1 * x2

                # Derivatives of loss function relative to the outputs of previous node
                d_layers[nodes_before_previous_layer + u1] += \
                    dy * (weights[WPERNODE*(nodes_before_current_layer + node) + 1] \
                        + weights[WPERNODE*(nodes_before_current_layer + node) + 3] * x2)
                d_layers[nodes_before_previous_layer + u2] += \
                    dy * (weights[WPERNODE*(nodes_before_current_layer + node) + 2] \
                        + weights[WPERNODE*(nodes_before_current_layer + node) + 3] * x1)
                d_layers[nodes_before_current_layer + node] = 0

            nodes_before_current_layer = nodes_before_previous_layer

    nodes_before_current_layer = 0

    # Update all weights
    #contador = 0 # Debug
    for layer in range(1, NLAYERS):
        current_layer_size = layer_sizes[layer]
        #print("layer", layer)

        nodes_before_current_layer += layer_sizes[layer - 1]
        for node in range(current_layer_size):
            for weight in range(WPERNODE):
                a = activation(weights[WPERNODE*(nodes_before_current_layer + node) + weight], \
                               d_weights[WPERNODE*(nodes_before_current_layer + node) + weight], \
                               rel_err/NBATCH) / 2 # Why divided by 2?

                weights[WPERNODE*(nodes_before_current_layer + node) + weight] -= a
                d_weights[WPERNODE*(nodes_before_current_layer + node) + weight] = 0


    np.savetxt("weights_during_regression", weights)
    #print("end of batch", batch) #Debug



if __name__ == "__main__" :
    # We store the layers' sizes (number of nodes) with this array
    layer_sizes = np.array([])
    # The array "layers" contains each node output
    layers = np.zeros(INSIZE).astype(np.float64)
    # The array "weights" contains all the weights for all nodes
    # weights = np.zeros(INSIZE*WPERNODE).astype(np.float64)
    weights = (np.random.random(INSIZE*WPERNODE)-.5).astype(np.float64) # Off for debug


    # The array "d_layers" contains the gradients relative to the outputs of the nodes
    d_layers = np.zeros(INSIZE).astype(np.float64)
    # The array "d_weights" contains the gradients relative to the weights of the nodes
    d_weights = np.zeros(INSIZE * WPERNODE).astype(np.float64)

    # Protein Network architecture, similar to a inverse binary tree.
    # Each node at each layer is connected to 2 nodes from previous layers
    # The network is built inversely, from output to input
    current_layer_size = OUTSIZE
    for layer in reversed(range(NLAYERS)):
        layer_sizes = np.hstack((np.array([current_layer_size]), layer_sizes)).astype(np.int32)
        current_layer_size = min(2*current_layer_size, MAXSIZE)
        if layer == 1: current_layer_size = INSIZE


    # It should return a vector with the sizes (number of nodes per layer) of all layers.
    print("Architecture")
    print(layer_sizes)

    # Layers 1 to "NLAYERS - 1" contains "layer_sizes[layer]" nodes per layer
    # Each "layers" cell contains the output for each node from each layer
    # Weights for all nodes are initialized with random values from uniform distribution U[-0.5, +0.5]
    for layer in range(1, NLAYERS):
        layers = np.hstack((layers, np.zeros(layer_sizes[layer]))).astype(np.float64)
        weights = np.hstack((weights, np.random.random(layer_sizes[layer]*WPERNODE) - .5)).astype(np.float64)
        d_layers = np.hstack((d_layers, np.zeros(layer_sizes[layer]))).astype(np.float64)
        d_weights = np.hstack((d_weights, np.zeros(layer_sizes[layer]*WPERNODE))).astype(np.float64)

    #input("Press Enter to continue...")
    np.savetxt("weights_before_regression", weights)

    #x_dom = np.arange(0,DOMAIN,BA).astype(np.float64)
    #y_dom = np.arange(0,DOMAIN,BA).astype(np.float64)
    x_dom = np.arange(-DOMAIN,DOMAIN,BA).astype(np.float64)
    y_dom = np.arange(-DOMAIN,DOMAIN,BA).astype(np.float64)

    z1 = np.array([])

    # Plots the original function
    fplotprep(x_dom, y_dom, z1)

    keep = True
    count = 0
    while( True ):
        count += 1

        seed = int(time.monotonic())
        regression(layers, weights, d_layers, d_weights, layer_sizes)
        if count % 100 == 0: print("count", count)
        # Plots the function by regression with the Protein Network
        if count % 1000 == 0:
            z1 = np.array([])
            fplotprep(x_dom, y_dom, z1)

            # Plots the function regressed by the Protein Network
            z2 = np.array([])
            nplotprep(x_dom, y_dom, z2, layers, weights, layer_sizes)

            np.savetxt("weights_after_regression", weights)
