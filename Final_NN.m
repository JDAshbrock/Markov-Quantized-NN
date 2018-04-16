clear all
close all
clc

PictureHeight=28;
PictureWidth = 28;
scale=16;

InputSize = PictureHeight*PictureWidth+1; % +1 for the bias
Layer1Size = 20;
NumberOfCategories = 10; % 1 per digit from 0-9
alpha = 1; %Scaling parameter so as to interpret gradients as probabilities
BaseRate=0.1;
ImageSet = LoadMNISTImages('train-images.idx3-ubyte');
Labels = LoadMNISTLabels('train-labels.idx1-ubyte');
TestImages = LoadMNISTImages('t10k-images.idx3-ubyte');
TestLabels = LoadMNISTLabels('t10k-labels.idx1-ubyte');
TestSize = 5000;
looplength=60000; %Control the length of the training loop - 60000 max.
TotalWeights=InputSize*Layer1Size + (Layer1Size+1)*NumberOfCategories;

Layer1Weights = zeros(Layer1Size,InputSize); %Layer1(i,j) contains the weight from input j into neuron input
FinalWeights=zeros(NumberOfCategories,Layer1Size+1); %FinalWeights(i,j) contains weight from neuron j into category i

%The following function implements the point-wise arctan needed in the feed-forward step



%Initialize Layer 1 weights 
for i=1:Layer1Size
	for j=1:InputSize
		Layer1Weights(i,j)=0.5*rand-0.25;       
	end
end
%Initialize Final layer weights
for i=1:NumberOfCategories
	for j=1:Layer1Size+1
        	FinalWeights(i,j)=0.5*rand-0.25;
	end
end

Layer1Outputs = zeros(1,Layer1Size); %Layer1Outputs(i) contains the output form neuron i, layer 1
FinalOutputs = zeros(1,NumberOfCategories); %FinalOutputs(i) contains the output corresponding to category i


% countoff=zeros(1,looplength);
% counton = zeros(1,looplength);
% oversat=zeros(1,looplength);
% undersat=zeros(1,looplength);
sup=0;
decay=1;
numchanged=zeros(1,looplength);
PartialAcc = zeros(1,floor(looplength/5000)); testnum=0; %These variables are used for intermittent accuracy test
for l=1:looplength%Training Loop
    decay=l/3000+1;
    LearningRate=BaseRate/decay;
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
    %Update
    Layer1Weights = Layer1Weights-LearningRate.*Layer1Gradient;
    FinalWeights = FinalWeights-LearningRate.*FinalGradient;
    sup1 = max(max(Layer1Gradient));
    sup2 =max(max(FinalGradient));
    sup=max([sup1,sup2,sup]); % This code tracks the size of the largest gradient value
    
    if(rem(l,1000)==1)
        l
    end
    % Test intermittently throughout training
    if(rem(l,1000)==0)
        counter=0;
        testnum = testnum+1;
        for i=1:TestSize
        Im = horzcat(TestImages(:,i)',1)';
        TrainingLabel = TestLabels(i);
        Layer1Outputs = horzcat(pt_atan(Layer1Weights*Im),1)';
        FinalOutputs = FinalWeights*Layer1Outputs;
    	Class = find(FinalOutputs == max(FinalOutputs))-1;% the -1 because index 1 corresponds to 
    	% classification as a 0
    	if(Class==TrainingLabel) %#ok<ALIGN>
        	counter = counter+1;
        end
        end
        acc=counter/TestSize
        PartialAcc(testnum)=acc;
    end %End test
end %End train loop
    
    



%Final diagnostic plots
Accuracy = counter/TestSize; %Final Accuracy
DisplayImage(Layer1Weights(1,:)); 
DisplayImage(Layer1Weights(2,:)); 
DisplayImage(Layer1Weights(3,:)); 
