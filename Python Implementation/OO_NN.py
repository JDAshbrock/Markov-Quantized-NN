'''Organization:
    - Partial derivatives and weight updates happen at the neuron layer
    - The layer class handle passing of information between layers
    - The NN class handles the training and prediction functions
    
    
    
    '''
import random
import math
import time
import numpy
import matplotlib as plt
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#The data is mnist.train.images which is (55,000x784) and mnist.train.labels which is (55,0000x10)

random.seed(1)
LearningRate = 0.005


class Neuron(object):
    # This class contains the set of weights INTO the neuron and the bias INTO the neuron
    # The nruon has attributes: propagate (the signal), updateWeights, updateBias, and computeOutPartial.
    def __init__(self, size, NeuronNumber, sd):
        self.Weights = []
        for i in range(0, size):
            self.Weights.append(random.gauss(0, 0.1))
        self.Bias = random.gauss(0, 0.1)
        self.Size = size
        self.ID = NeuronNumber
        self.Output=0
        self.OutPartial=0

    def propagate(self, prevLayer):
        self.Output = self.Bias
        for i in range(0, prevLayer.LayerSize):
            self.Output += self.Weights[i] * prevLayer.Outputs[i]
        self.UnfilteredOutput=self.Output
        if self.UnfilteredOutput <0:
            self.Output =0
        #print("ID: " + str(self.ID) +" Output: " + str(self.Output) +"\n")

    def update_Weights(self, prevLayer):
        # prevLayer is the previous "Layer" object
        if self.Output > 0:
            for i in range(0, prevLayer.LayerSize):
                self.Weights[i] -= LearningRate * self.OutPartial * prevLayer.Outputs[i]

    def update_Bias(self):
        if self.Output > 0:
            self.Bias -=self.OutPartial * LearningRate
    
    def compute_Out_Partial(self, nextLayer):
        # next Layer is the next "Layer" object
        self.OutPartial = 0
        for i in range(0, nextLayer.LayerSize):
            if nextLayer.Nodes[i].Output>0:
                self.OutPartial += nextLayer.OutPartials[i] * nextLayer.Nodes[i].Weights[self.ID]


class InputLayer(object):
    def __init__(self, inputVec):
        self.Outputs=inputVec
        self.LayerSize=len(inputVec)
    
    def new_input(self,next_image):
        self.Outputs=next_image
        self.LayerSize=len(next_image)
        
    def display(self):
        self.mat = numpy.mat(self.Outputs)
        self.mat = self.mat.reshape(28,28)
        plt.pyplot.figure()
        imgplot = plt.pyplot.imshow(self.mat)
        
    
    
class HiddenLayer(object):
    
    def __init__(self, size, prevSize):
    # This layer contains all weights out of the previous layer
        self.Nodes = []
        self.OutPartials=[]
        self.Outputs=[]
        self.sd=2/math.sqrt(prevSize)
        for i in range(0, size):
            self.Nodes.append(Neuron(prevSize, i, self.sd))
            self.OutPartials.append(0)
            self.Outputs.append(0)
        self.LayerSize = size      

    def propagate(self, prevLayer):
        for i in range(0, self.LayerSize):
            self.Nodes[i].propagate(prevLayer)
            self.Outputs[i]=self.Nodes[i].Output
            
    def output_Partials(self, nextLayer):
        for i in range(0, self.LayerSize):
            self.Nodes[i].compute_Out_Partial(nextLayer)
            
    def update(self, prevLayer):
        for i in range(0, self.LayerSize):
            self.Nodes[i].update_Weights(prevLayer)
            self.Nodes[i].update_Bias()

class LastHiddenLayer(HiddenLayer):
    
    def output_Partials(self, nextLayer):
        for i in range(0, self.LayerSize):
            self.OutPartials[i]=-nextLayer.Nodes[nextLayer.correctClass].Weights[i]
            for j in range(0, nextLayer.LayerSize):
                self.OutPartials[i]+=nextLayer.Output[j]*nextLayer.Nodes[j].Weights[i]
                    
class OutputLayer(object):
    
    def __init__(self, size, prevSize, correctClass=0):
    # This layer contains all weights out of the previous layer
        self.Nodes = []
        self.OutPartials =[]
        self.expOut=[]
        self.Output=[]
        self.sd=2/math.sqrt(prevSize)
        for i in range(0, size):
            self.Nodes.append(Neuron(prevSize, i, self.sd))
            self.OutPartials.append(0)
            self.expOut.append(0)
            self.Output.append(0)
        self.LayerSize = size
        self.correctClass=correctClass
        
    def propagate(self, prevLayer):
        # Here we use the Unfiltered output which does not contain the bias or the relu non-linearity
        for i in range(0, self.LayerSize):
            self.Nodes[i].propagate(prevLayer)
        self.total = 0
        for i in range(0, self.LayerSize):
            try:
                self.expOut[i]=math.exp(self.Nodes[i].UnfilteredOutput)
            except: 
                print("Overflow")
                time.sleep(10)
            self.total += self.expOut[i]
        self.curmax = 0
        for i in range(0, self.LayerSize):
            self.Output[i]=self.expOut[i] / self.total
            if self.Output[i] > self.curmax:
                self.classification = i
                self.curmax = self.Output[i]
        return self.classification

    def update(self, prevLayer):
        for i in range(0, self.LayerSize):
            for j in range(0, prevLayer.LayerSize):
                # Weight from neuron j into node i
                if self.correctClass == i:
                    self.Nodes[i].Weights[j]-=LearningRate*prevLayer.Nodes[j].Output*(self.Output[i]-1)
                else:
                    self.Nodes[i].Weights[j]-=LearningRate*prevLayer.Nodes[j].Output*(self.Output[i])


class FFNN(object):

    def __init__(self, num_hidden_layers, HL_sizes, num_cats, input_size):
        if num_hidden_layers >2:
            self.Hidden_Layers = [HiddenLayer(HL_sizes,input_size)]
            for i in range(0, num_hidden_layers-2):
                self.Hidden_Layers.append(HiddenLayer(HL_sizes,HL_sizes))
            self.Hidden_Layers.append(LastHiddenLayer(HL_sizes,HL_sizes))
        else:
            self.Hidden_Layers=[LastHiddenLayer(HL_sizes,input_size)]
        self.Output_Layer=OutputLayer(num_cats,HL_sizes)
        self.num_hidden=num_hidden_layers
        self.input=InputLayer([])
        self.predicted_class=0

    def predict(self):
        self.Hidden_Layers[0].propagate(self.input)
        for i in range(1,self.num_hidden):
            self.Hidden_Layers[i].propagate(self.Hidden_Layers[i-1])
        self.predicted_class = self.Output_Layer.propagate(self.Hidden_Layers[-1])
        

    def train(self, its):
        for i in range(0, its):
            self.predict()     
            self.Output_Layer.update(self.Hidden_Layers[-1])
            next_layer = self.Output_Layer
            for j in range(0, self.num_hidden-1):
                index = self.num_hidden-j-1
                self.Hidden_Layers[index].output_Partials(next_layer)
                self.Hidden_Layers[index].update(self.Hidden_Layers[index-1])
                next_layer = self.Hidden_Layers[index]
            self.Hidden_Layers[0].output_Partials(next_layer)
            self.Hidden_Layers[0].update(self.input)
            

            
    
    def SGD(self, input_list, class_list, iterations):
        self.count=0
        for i in range(0, iterations):
            if i%1000==999:
                print("Correct: " + str(self.count))
                self.count=0
            self.input.new_input(input_list[i,:])
            #self.input.display()
            for j in range(0, 10):
                if class_list[i,j]!=0:
                    self.Output_Layer.correctClass=j
            self.input.new_input(input_list[i])
            self.train(1)
            if self.predicted_class==self.Output_Layer.correctClass:
                self.count+=1
        print("Computed Accuracy: " + str(self.count/iterations))
    

            
my_net = FFNN(3,12,10,784)
my_net.SGD(mnist.train.images,mnist.train.labels,60000)


                
            
        
        
        
        

        