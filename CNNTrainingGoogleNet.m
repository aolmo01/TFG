clear all; close all;

% https://www.kaggle.com/datasets/gpiosenka/70-dog-breedsimage-data-set

% The following paths must be changed to those of the DataSet that will be used for training
TrainingDogs = "C:\Users\Alberto\OneDrive\Documentos\TFG\data\GoogleNet\train";
ValidationDogs = "C:\Users\Alberto\OneDrive\Documentos\TFG\data\GoogleNet\valid";
TestDogs = "C:\Users\Alberto\OneDrive\Documentos\TFG\data\GoogleNet\test";

imdsTrain = imageDatastore(TrainingDogs,...
    'IncludeSubfolders',true,...
    'LabelSource','foldernames');

imdsValidation = imageDatastore(ValidationDogs,...
    'IncludeSubfolders',true,...
    'LabelSource','foldernames');

imdsTest = imageDatastore(TestDogs,...
    'IncludeSubfolders',true,...
    'LabelSource','foldernames');

%[imdsTrain,imdsValidation] = splitEachLabel(imdsTrain,0.7,'randomized');

%if not(isfile('netTransferAlexNet.mat'))
if not(isfile('netTransferGoogleNet.mat'))

  numTrainImages = numel(imdsTrain.Labels);
  idx = randperm(numTrainImages,16);
  figure
  for i = 1:16
    subplot(4,4,i)
    I = readimage(imdsTrain,idx(i));
    imshow(I)
  end

  % load netlayer
  %net = alexnet;
  net = googlenet;
  % net = resnet101;

    net.Layers(1)
    inputSize = net.Layers(1).InputSize;
    
    analyzeNetwork(net)

    if isa(net,'SeriesNetwork') 
      lgraph = layerGraph(net.Layers); 
    else
      lgraph = layerGraph(net);
    end 
    
    [learnableLayer,classLayer] = findLayersToReplace(lgraph);
    
    numClasses = numel(categories(imdsTrain.Labels));
    
    if isa(learnableLayer,'nnet.cnn.layer.FullyConnectedLayer')
        newLearnableLayer = fullyConnectedLayer(numClasses, ...
            'Name','new_fc', ...
            'WeightLearnRateFactor',10, ...
            'BiasLearnRateFactor',10);
        
    elseif isa(learnableLayer,'nnet.cnn.layer.Convolution2DLayer')
        newLearnableLayer = convolution2dLayer(1,numClasses, ...
            'Name','new_conv', ...
            'WeightLearnRateFactor',10, ...
            'BiasLearnRateFactor',10);
    end
    
    lgraph = replaceLayer(lgraph,learnableLayer.Name,newLearnableLayer);
    
    newClassLayer = classificationLayer('Name','new_classoutput');
    lgraph = replaceLayer(lgraph,classLayer.Name,newClassLayer);
    
    figure('Units','normalized','Position',[0.3 0.3 0.4 0.4]);
    plot(lgraph)
    ylim([0,10])
    
    layers = lgraph.Layers;
    connections = lgraph.Connections;
    
    layers(1:10) = freezeWeights(layers(1:10));
    lgraph = createLgraphUsingConnections(layers,connections);
    
    analyzeNetwork(lgraph)
    
    pixelRange = [-30 30];
    scaleRange = [0.9 1.1];
    imageAugmenter = imageDataAugmenter( ...
        'RandXReflection',false, ...
        'RandXTranslation',pixelRange, ...
        'RandYTranslation',pixelRange, ...
        'RandXScale',scaleRange, ...
        'RandYScale',scaleRange);
    
    augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
        'DataAugmentation',imageAugmenter);
    [dataTrain,infoTrain]=read(augimdsTrain);
    
    augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation, ...
        'DataAugmentation',imageAugmenter);
    [dataValidation,infoValidation]=read(augimdsValidation);
    
    numaugsValidationImages = augimdsValidation.NumObservations;
    idx = randperm(numaugsValidationImages,16);
    
    figure
    for i = 1:16
        subplot(4,4,i)
        I = imread(augimdsValidation.Files{idx(i),1});
        imshow(I)
    end
    
    options = trainingOptions('adam', ...
        'MiniBatchSize',10, ...
        'MaxEpochs',8, ...
        'InitialLearnRate',1e-4, ...
        'ValidationData',augimdsValidation, ...
        'ValidationFrequency',3, ...
        'ValidationPatience',Inf, ...
        'Verbose',false, ...
        'Plots','training-progress');
    
    % CategTrain = categories(augimdsTrain.Labels);
    % CategValidation = categories(augimdsValidation.Labels);
    
    netTransfer = trainNetwork(augimdsTrain,lgraph,options);
    save netTransferGoogleNet netTransfer
    %save netTransferAlexNet netTransfer
else
  load netTransferGoogleNet
  %load netTransferAlexNet
end


%% Validación del proceso de entrenamiento

[YTestPred,probs] = classify(netTransfer,imdsTest);
testAccuracy = mean(YTestPred == imdsTest.Labels)
testError = mean(YTestPred ~= imdsTest.Labels)

YTrainPred = classify(netTransfer,imdsTrain);
trainError = mean(YTrainPred ~= imdsTrain.Labels);
disp("Error Entrenamiento: " + trainError*100 + "%")
disp("Error Test: " + testError*100 + "%")

%% Plot the confusion matrix. Display the precision and recall for each class by using column and row summaries. Sort the classes of the confusion matrix. The largest confusion is between unknown words and commands, up and off, down and no, and go and no.
figure('Units','normalized','Position',[0.2 0.2 0.5 0.5]);
cm = confusionchart(YTestPred,imdsTest.Labels, ...
    'ColumnSummary','column-normalized', ...
    'RowSummary','row-normalized');
cm.Title = 'Matriz Confusion Test GoogleNet';
%sortClasses(cm)

idx = randperm(numel(imdsTest.Files),8);
figure
for i = 1:8
    subplot(4,2,i)
    I = readimage(imdsTest,idx(i));
    imshow(I)
    label = YTestPred(idx(i));
    title(string(label) + ", " + num2str(100*max(probs(idx(i),:)),3) + "%");
end

%% Visualización de los pesos

% Get the network weights for the second convolutional layer
w1 = netTransfer.Layers(2).Weights;

% Scale and resize the weights for visualization
w1 = mat2gray(w1);
w1 = imresize(w1,5);

% Display a montage of network weights. There are 96 individual sets of
% weights in the first layer.
figure
montage(w1)
title('Pesos primera capa convolucional')


