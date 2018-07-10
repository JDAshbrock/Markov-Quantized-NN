# This program is an implementation of the simple neural network in an object-oriented format.
#The basic objects are neurons. Then the neuron objects are organized into hidden layer objects.
# Finally, the network contains a list of layers as objects
# Currently, the following simplifications have been made: Only ReLU nonlinearity, all hidden layers are the same size


# CURRENT UPDATE: Finished writing neuron and hidden layer. Neither are really tested
# Need to include a Boolean value in hidden layer class to say if it is the last hidden layer. This is necessary because
# There is a different formula for the output partial derivatives

import random
import math
import numpy as np

learningRate = 0.1


class Neuron:
    # This class contains the set of weights INTO the neuron and the bias INTO the neuron
    # The nruon has attributes: propagate (the signal), updateWeights, updateBias, and computeOutPartial.
    def __init__(self, size):
        self.Weights = []
        for i in range(0, size):
            self.Weights.append(random.gauss(0, 0.25))
        self.Bias = random.gauss(0, 0.25)
        self.Size = size

    def propagate(self, input):
        # filter is a boolean value telling us whether or not to filter the result
        self.Output = 0
        for i in range(0, len(input)):
            self.Output += self.Weights[i] * input[i]

        self.UnfilteredOut = self.Output
        self.Output += self.Bias
        if self.Output <0:
            self.Output =0

    def updateWeights(self, previousLayerOutputs, outPartial):
        if self.Output > 0:
            for i in range(0, len(previousLayerOutputs)):
                self.Weights[i] = self.Weights[i] - LearningRate * outPartial * previousLayerOutputs[i]

    def updateBias(self, outPartial):
        if self.Output > 0:
            self.Bias = self.Bias - outPartial * LearningRate

    def computeOutPartial(self, nextPartials, weightsOut, nextOutput):
        # This will compute the partial derivative of the error with respect to the output of this neuron
        # nextPartials is the error partial w.r.t. the next layer outputs
        # weightsOut are the weights out of this neuron
        # nextOutput is the Output of the neurons in the next layer
        self.OutPartial = 0
        for i in range(0, len(weightsOut)):
            if nextOutput[i] > 0:
                self.OutPartial += nextPartials[i] * weightsOut[i]


class HiddenLayer:
    def __init__(self, size, prevSize, LastLayer):
        # LastLayer is a Boolean which is true if this is the last hidden layer
        self.Nodes = []
        for i in range(0, size):
            self.Nodes.append(Neuron(prevSize))
        self.Size = size
        self.LL=LastLayer

    def propagate(self, prevOutput):
        self.OutputVec = []
        for i in range(0, self.Size):
            self.Nodes[i].propagate(prevOutput)
            self.OutputVec.append(self.Nodes[i].Output)

    def update(self, prevOutput, outPartial):
        for i in range(0, self.Size):
            self.Nodes[i].updateWeights(prevOutput, outPartial)
            self.Nodes[i].updateBias(outPartial)

    def computeStatePartials(self, nextPartials, weightsOut, nextOutput):
        # Here, nextPartials is a vector of partial derivatives from the next layer,
        # weightsOut is a numpy matrix of weights out of the current layer. Entry [i,j] contains weight out of j, into i
        # nextOutput is the vector containing the outputs of the next layer neurons
        for i in range(0, self.Size):
            #Need weights OUT of neuron i
            self.weightVec = weightsOut
            self.Nodes[i].ComputeOutPartial(nextPartials, weightsOut[:,i], nextOutput)


class OutputLayer:
    def __init__(self, numcats, prevSize):
        self.Nodes = []
        for i in range(0, numcats):
            self.Nodes.append(Neuron(prevSize))
        self.numCats = numcats
        self.prevSize = prevSize

    def classify(self, input):
        # Here we use the Unfiltered output which does not contain the bias or the relu non-linearity
        for i in range(0, self.numCats):
            self.Nodes[i].propagate(input)
        self.expOut = []
        self.total = 0
        for i in range(0, self.numCats):
            self.expOut.append(math.exp(self.Nodes[i].UnfilteredOut))
            self.total += self.expOut[i]
        self.Output = []
        self.curmax = 0
        for i in range(0, self.numCats):
            self.Output.append(self.expOut[i] / self.total)
            if self.Output[i] > self.curmax:
                self.classification = i
                curmax = self.Output[i]
        return self.classification

    def update(self, prevOutput, correctClass):
        for i in range(0, self.numCats):
            for j in range(0, self.prevSize):
                # Weight from neuron j into node i
                if correctClass == i:
                    self.Nodes[i].Weights[j]-=learningRate*prevOutput[j]*(self.Output[i]-1)
                else:
                    self.Nodes[i].Weights[j]-=learningRate*prevOutput[j]*self.Output[i]


class FFNN:

    def __init__(selfself, hiddenlayers, layersize, cats, inputsize):
        # method creates a list of hidden layer objects with a final output layer included by default
        self.Layers = [HiddenLayer(layersize,inputsize)]




