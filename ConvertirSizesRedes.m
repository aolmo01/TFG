
%% Redimensionado de las imágenes
% Dimensión GoogleNet 224x224x3
% Dimensión AlexNet 227x227x3
% The following paths must be changed to those of the DataSet that will be used for training
TrainingDogs = "C:\Users\Alberto\OneDrive\Documentos\TFG\data\AlexNet\train";
ValidationDogs = "C:\Users\Alberto\OneDrive\Documentos\TFG\data\AlexNet\valid";
TestDogs = "C:\Users\Alberto\OneDrive\Documentos\TFG\data\AlexNet\test";

%imds = imageDatastore(TrainingDogs,...
%    'IncludeSubfolders',true,...
%    'LabelSource','foldernames');

%imds = imageDatastore(ValidationDogs,...
%    'IncludeSubfolders',true,...
%    'LabelSource','foldernames');

imds = imageDatastore(TrainingDogs,...
    'IncludeSubfolders',true,...
    'LabelSource','foldernames');

idx = size(imds.Files,1);

for i=1:1:idx
  fichero = cell2mat(imds.Files(i));
  Img = imread(fichero);
  
  % GoogleNet
  %I = imresize(Img, [224 224]);
  
  % AlexNet
  I = imresize(Img, [227 227]);
  
  imwrite(I,fichero);
end
