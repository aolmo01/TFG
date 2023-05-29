function classifyImage(image, cnn)
    if cnn == "GoogleNet"
        load('netTransferGoogleNet.mat');
    elseif cnn == "AlexNet"
        load('netTransferAlexNet.mat');
    end

    Img = imread(image);
    [YTestPred,probs] = classify(netTransfer,Img);

    imshow(Img)
    title(string(YTestPred) + ", " + num2str(100*max(probs),3) + "%");
end