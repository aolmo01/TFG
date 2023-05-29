function resizeImage(image, cnn)
    %% Redimensionado de las imágenes
    % Dimensión GoogleNet 224x224x3
    % Dimensión AlexNet 227x227x3
    Img = imread(image);

    if cnn == "GoogleNet"
        I = imresize(Img, [224 224]); 
    elseif cnn == "AlexNet"
        I = imresize(Img, [227 227]);
    end
    
    imwrite(I,image);
end