function [features] = hw_features(data, windowSize, meanWindow)
    numSamples = size(data,1);
    numWin = floor(numSamples/windowSize);
    numChannels = size(data,2);

    val = zeros(numChannels,numWin);
    
    buffer = zeros(meanWindow,numChannels);
    means = zeros(1,numChannels);
    idx = 1;
    meanIdx = 1;
    feat = zeros(1,numChannels);
    w = 1;
    for i = 1:numSamples
        
        means = means - buffer(meanIdx,:);
        means = means + data(i,:);
        feat = feat + abs(data(i,:) - floor(means./meanWindow));
        buffer(meanIdx,:) = data(i,:);
        if idx == windowSize
            idx = 1;
            val(:,w) = floor(feat/32)';
            feat = zeros(1,64);
            w = w + 1;
        else
            idx = idx + 1;
        end

        if meanIdx == meanWindow
            meanIdx = 1;
        else
            meanIdx = meanIdx + 1;
        end
    end

    % for i = 1:numWin
    %     featLabel(i) = mode(label((1:windowSize)+(i-1)*windowSize));
    %     for ch = 1:numChannels
    %         val(ch,i) = featureFunc(data((1:windowSize)+(i-1)*windowSize,ch));
    %     end
    % end
    val = val(:,2:end);
%     val(val > 63) = 63;
%     featLabel = featLabel(2:end);
%     features(g,tr).values = val';
%     features(g,tr).label = featLabel;
    features = val;
end