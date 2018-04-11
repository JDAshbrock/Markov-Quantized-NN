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
BaseRate=10;
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



%Initialize Layer 1 weights to a Bernoulli r.v. with p=0.5
for i=1:Layer1Size
	for j=1:InputSize
        rv = randi([-scale,scale]);
        Layer1Weights(i,j)=rv/scale;
	end
end
%Initialize Final layer weights as a Bernoulli r.v. with p=0.5
for i=1:NumberOfCategories
	for j=1:Layer1Size+1
        rv=randi([-scale,scale]);
        FinalWeights(i,j)=rv/scale;
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
    decay=floor(l/10000)+1;
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
            rv=rand/LearningRate;
            if(j~= Layer1Size+1)
                NeuralOutputGradients(j)=NeuralOutputGradients(j)+FinalGradient(i,j)*(FinalWeights(i,j)/Layer1Outputs(j));
            end
            delta=0;
            if(FinalGradient(i,j)<0 && rv<((-1)*FinalGradient(i,j)))
                delta=1;
            elseif(FinalGradient(i,j)>0 && rv<FinalGradient(i,j))
                delta=-1;
            end 
           FinalWeights(i,j)=FinalWeights(i,j)+delta/scale;
           if(FinalWeights(i,j)>1)
               FinalWeights(i,j)=1;
               delta=0;
           elseif(FinalWeights(i,j)<-1)
               FinalWeights(i,j)=-1;
               delta=0;
           end
           numchanged(l)=numchanged(l)+abs(delta);
        end
	end
	% This loop iterates over all weights in the initial Layer
	% It first computes the gradient with respect to each of the weights
	% Then it updates the weights as described in the algorithm
	for i=1:Layer1Size %#ok<ALIGN>
		for j=1:InputSize %#ok<ALIGN>
			temp = Layer1Weights(i,:)*Im;
			Layer1Gradient(i,j)= NeuralOutputGradients(i)*(1/(pi+pi*temp*temp))*Im(j);
            rv=rand/LearningRate; 
            delta=0;
            if(Layer1Gradient(i,j) <0 && rv<((-1)*Layer1Gradient(i,j)))
                delta=1;
            elseif(Layer1Gradient(i,j)>0 && rv<Layer1Gradient(i,j))
               delta=-1;
            end
            Layer1Weights(i,j)=Layer1Weights(i,j)+delta/scale;
            if(Layer1Weights(i,j)>1)
                Layer1Weights(i,j)=1;
                delta=0;
            elseif(Layer1Weights(i,j)<-1)
                Layer1Weights(i,j)=-1;
                delta=0;
            end
            numchanged(l)=numchanged(l)+abs(delta);
        end
    end
    sup1 = max(max(Layer1Gradient));
    sup2 =max(max(FinalGradient));
    sup=max([sup1,sup2,sup]); % This code tracks the size of the largest gradient value
    if(rem(l,1000)==1)
        l
    end
    % Test intermittently throughout training
    if(rem(l,5000)==0)
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
    
    


%The following code smooths the "parameters updated" by taking a running
%average of the previous "smoothfactor" parameter updates
smoothfactor=100;
smoothedchanged =zeros(1,l-smoothfactor);
for i=smoothfactor:length(numchanged)
    temp=0;
    for j=1:smoothfactor-1
        temp=temp+numchanged(i-j+1);
    end
    smoothedchanged(i)=temp;
end
len=length(smoothedchanged);

%Final diagnostic plots
Accuracy = counter/TestSize; %Final Accuracy
plot(1:len, smoothedchanged);
plot(1:l,numchanged(1:l));
DisplayImage(Layer1Weights(1,:)); 
DisplayImage(Layer1Weights(2,:)); 
DisplayImage(Layer1Weights(3,:)); 
