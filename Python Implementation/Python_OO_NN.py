# This program is an object-oriented version of the neural network algorithm.
# The following object types are defined: Neuron, Hidden Layer, Neural Network
# Each object above requires use of the previous one
# Neurons have the following attributes: weights, bias, and state


''' This is entirely incomplete, I just simply ran out of steam doing it
The only thing I have not even begun is the output partials in the last hidden layer
Nothing has been tested except for the propagate and single neuron weight update
still needs a class for the full network 
still needs to be able to import the data
still needs to include a training loop'''


import numpy as np
import random
import math

learningRate = 0.1

class Neuron:
	#This class contains the set of weights INTO the neuron and the bias INTO the neuron
	# The nruon has attributes: propagate (the signal), updateWeights, updateBias, and computeOutPartial.
	def __init__(self, size):
		self.Weights =[]
		for i in range(0,size):
			self.Weights.append(random.gauss(0, 0.25))
		self.Bias = random.gauss(0,0.25)
		self.Size = size
	
	def propagate(self, input):
		# filter is a boolean value telling us whether or not to filter the result 
		self.Output=0
		for i in range(0, len(input)):
			self.Output += self.Weights[i]*input[i]
		
		self.UnfilteredOut = self.Output
		self.Output += self.Bias
		
	def updateWeights(self, previousLayerOutputs, outPartial):
		if self.Output >0:
			for i in range(0, len(previousLayerOutputs)):
				self.Weights[i] = self.Weights[i] - LearningRate*outPartial*previousLayerOutputs[i]
	
	def updateBias(self, outPartial):
		if self.Output >0:
			self.Bias = self.Bias - outPartial*LearningRate
		
	def computeOutPartial(self, nextPartials, weightsOut, nextOutput)
		self.OutPartial=0
		for i in range(0, len(weightsOut)):
			if nextPreFilter[i] >0:
				self.OutPartial += nextPartials[i]*weightsOut[i]
		

class HiddenLayer:
	def __init__(self, size, prevSize, LastLayer):
		# LastLayer is a Boolean which is true if this is the last hidden layer
		self.Nodes=[]
		for i in range(0,size):
			self.Nodes.append(Neuron(prevSize))
		self.Size = size
		self.LL = LastLayer
	
	def propagate(self, prevOutput):
		self.OutputVec =[]
		for i in range(0,self.Size):
			self.Nodes[i].propagate(prevOutput)
			self.OutputVec.append(self.Nodes[i].Output)
	
	def update(self, prevOutput, outPartial):
		for i in range(0,self.Size):
			self.Nodes[i].updateWeights(prevOutput, outPartial)
			self.Nodes[i].updateBias(outPartial)
	
	def computeLayerOutPartials(self, nextPartials, weightsOut,nextOutput, correctClass):
		#Here, nextPartials is a vector of partial derivatives from the next layer, nextOutput is the vector output of next layer
		# and weightsOut is a matrix where weightsOut[i,j] contains the weight out of neuron i into neuron join
		for i in range(0, self.Size):
			if not self.LL:
				dummy = weightsOut[ , i]
				self.Nodes[i].computeOutPartial(nextPartials, dummy, nextOutput)
			else:
				self.Nodes[i].OutPartial = 
				
				

class OutputLayer:
	def __init__(self, numcats, prevSize):
		self.Nodes=[]
		for i in range(0,numcats):
			self.Nodes.append(Neuron(prevSize))
		self.numcats = numcats
	
	
	def classify(self, input):
		# Here we use the Unfiltered output which does not contain the bias or the relu non-linearity
		for i in range(0, self.numcats):
			self.Nodes[i].propagate(input)
		self.expOut=[]
		self.total = 0
		for i in range(0, self.numcats):
			self.expOut.append(math.exp(self.Nodes[i].UnfilteredOut))
			self.total += self.expOut[i]
		self.Output = []
		self.curmax = 0
		for i in range(0, self.numcats):
			self.Output.append(self.expOut[i]/self.total)
			if self.Output[i] > curmax
				self.classification = i
				curmax = self.Output[i]
		return self.classification
	
	def update(self, prevOutput)
		
		
	
	
		
	
	
			
		
			
		
		
		
		
	


neuron1 = Neuron(10)
neuron1.Bias
neuron1.Weights

		
		

		
	