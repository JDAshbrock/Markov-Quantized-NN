clear all
close all
clc

PictureHeight=28;
PictureWidth = 28;

InputSize = PictureHeight*PictureWidth+1; % +1 for the bias
Layer1Size = 20;
NumberOfCategories = 10; % 1 per digit from 0-9
alpha = 1; %Scaling parameter so as to interpret gradients as probabilities
LearningRate=0.05;
ImageSet = LoadMNISTImages('train-images.idx3-ubyte');
Labels = LoadMNISTLabels('train-labels.idx1-ubyte');
TestImages = LoadMNISTImages('t10k-images.idx3-ubyte');
TestLabels = LoadMNISTLabels('t10k-labels.idx1-ubyte');
TestSize = 10000;

Layer1Weights = zeros(Layer1Size,InputSize); %Layer1(i,j) contains the weight from input j into neuron input
FinalWeights=zeros(NumberOfCategories,Layer1Size+1); %FinalWeights(i,j) contains weight from neuron j into category i

%The following function implements the point-wise arctan needed in the feed-forward step



%Initialize Layer 1 weights to a Bernoulli r.v. with p=0.5
for i=1:Layer1Size
	for j=1:InputSize
		Layer1Weights(i,j) = 0.5*rand-0.25;
	end
end
%Initialize Final layer weights as a Bernoulli r.v. with p=0.5
for i=1:NumberOfCategories
	for j=1:Layer1Size
		FinalWeights(i,j)=0.5*rand-0.25;
	end
end

Layer1Outputs = zeros(1,Layer1Size); %Layer1Outputs(i) contains the output form neuron i, layer 1
FinalOutputs = zeros(1,NumberOfCategories); %FinalOutputs(i) contains the output corresponding to category i


for l=1:60000%Training Loop
    
	Im = horzcat(ImageSet(:,l)',1)';
    LabelVector=zeros(NumberOfCategories,1);
    ind = Labels(l)+1;
    LabelVector(Labels(l)+1)=1;
	%Feed Forward
	Layer1Outputs = horzcat(pt_atan(Layer1Weights*Im),1)'; %The concatenated one takes care of the bias
	FinalOutputs = FinalWeights*Layer1Outputs;
	Error = FinalOutputs - LabelVector;
	%Reset the gradient vectors to all zeros
	FinalGradient = zeros(NumberOfCategories,Layer1Size+1); 
	Layer1Gradient = zeros(Layer1Size,InputSize);
	NeuralOutputGradients = zeros(1, Layer1Size);
	% This loop iterates over all weights in the classification Layer
	% It first computes the gradient with respect to each of the weights
	% Then it updates the weights as described in the algorithm
	% We must store these weights because they are used in the gradient computation of layer1
	for i=1:NumberOfCategories
		for j=1:(Layer1Size+1) %#ok<ALIGN>
			temp = FinalWeights(i,:)*Layer1Outputs;
			FinalGradient(i,j) = 2*alpha*Error(i)*Layer1Outputs(j);
            if(j~= Layer1Size+1)
                NeuralOutputGradients(j)=NeuralOutputGradients(j)+FinalGradient(i,j)*(FinalWeights(i,j)/Layer1Outputs(j));
            end
        end
       
	end
	% This loop iterates over all weights in the initial Layer
	% It first computes the gradient with respect to each of the weights
	% Then it updates the weights as described in the algorithm
	for i=1:Layer1Size %#ok<ALIGN>
		for j=1:InputSize %#ok<ALIGN>
			temp = Layer1Weights(i,:)*Im;
			Layer1Gradient(i,j)= NeuralOutputGradients(i)*(1/(pi+pi*temp*temp))*Im(j);
        end
    end

    Layer1Weights = Layer1Weights -LearningRate.*Layer1Gradient;
    FinalWeights = FinalWeights - LearningRate.*FinalGradient;
    if(rem(l,1000)==1)
        l
    end
end

%Now we test the accuracy of the trained network
counter =0;
for i=1:TestSize
	Im = horzcat(TestImages(:,i)',1)';
	TrainingLabel = TestLabels(i);
	Layer1Outputs = horzcat(pt_atan(Layer1Weights*Im),1)';
	FinalOutputs = FinalWeights*Layer1Outputs;
	Class = find(FinalOutputs == max(FinalOutputs))-1;% the -1 because index 1 corresponds to 
	% classification as a 0
	if(Class==TrainingLabel)
		counter = counter+1;
	end
end
Accuracy = counter/TestSize;