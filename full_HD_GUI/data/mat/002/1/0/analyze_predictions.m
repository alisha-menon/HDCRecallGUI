close all
clear
clc

dict = [100:112, 201:208];

files = dir('*.mat');
correct_all = [];
predict_all = [];
distance_all = [];
for file = files'
    load(file.name);

    mask = (2^15).*ones(size(data));
    msbs = bitand(mask,double(data)) > 0;
    p = msbs(:,60:64);
    d = msbs(:,50:59);
    
    prediction = zeros(size(p,1),1);
    for i = 1:5
        prediction = (prediction * 2) + p(:,i);
    end
    
    prediction (prediction > 20) = 0;
    
    distance = zeros(size(p,1),1);
    for i = 1:10
        distance = (distance * 2) + d(:,i);
    end
    
    idx = find(gestLabel ~= 0);
    idx = (idx(1)+2000):(idx(end)-2000);
    correct = gestLabel(idx);
    prediction = prediction(idx)';
    distance = distance(idx);
    for i = 1:length(prediction)
        prediction(i) = dict(prediction(i) + 1);
    end
    correct_all = [correct_all correct];
    predict_all = [predict_all prediction];
    distance_all = [distance_all distance];
end

% notrest = find(correct_all ~= 100);
% correct_all = correct_all(notrest);
% predict_all = predict_all(notrest);

plot(correct_all)
hold on
plot(predict_all)

accuracy = sum(correct_all == predict_all)/length(predict_all)

gestDict = containers.Map ('KeyType','int32','ValueType','any');
gestDict(100) = 'Rest';
gestDict(101) = 'IndFlex';
gestDict(102) = 'IndExt';
gestDict(103) = 'MidFlex';
gestDict(104) = 'MidExt';
gestDict(105) = 'RinFlex';
gestDict(106) = 'RinExt';
gestDict(107) = 'PinFlex';
gestDict(108) = 'PinExt';
gestDict(109) = 'ThuFlex';
gestDict(110) = 'ThuExt';
gestDict(111) = 'ThuAdd';
gestDict(112) = 'ThuAbd';
gestDict(201) = 'One';
gestDict(202) = 'Two';
gestDict(203) = 'Three';
gestDict(204) = 'Four';
gestDict(205) = 'Five';
gestDict(206) = 'ThumbUp';
gestDict(207) = 'Fist';
gestDict(208) = 'Flat';

gestures = sort(cell2mat(gestDict.keys));
c = confusionmat(correct_all,predict_all,'Order',gestures);
temp = sum(c,2);
totalTrials = mean(temp(temp~=0));
c = c./totalTrials.*100;
c = round(c.*100)./100;

gestureLabels = cell(size(gestures));
for i = 1:length(gestures)
    gestureLabels{i} = gestDict(gestures(i));
end

baseSize = 700;
baseGest = 13;
scaledSize = round(baseSize*length(gestures)/baseGest);

f = figure;
set(f,'Position',[100 100 scaledSize 2*scaledSize])
imagesc(c)
colormap(flipud(gray(2048)))
axis square
xticks(1:length(gestures));
xticklabels(gestureLabels);
xtickangle(45);
yticks(1:length(gestures));
yticklabels(gestureLabels);
set(gca, 'FontSize', 14)

ylabel('Actual Gesture','FontSize',16,'FontWeight','bold')
xlabel('Predicted Gesture','FontSize',16,'FontWeight','bold')
% title(titleTxt,'FontSize',24,'FontWeight','bold')
caxis([0 100])
colorbar
dx = 0.3;
for i = 1:length(gestures)
    for j = 1:length(gestures)
        if c(i,j) > 0.01
            accTxt = num2str(c(i,j));
            dxScaled = dx*length(accTxt)/5;
            if i == j
                text(j-dxScaled-0.1,i, accTxt,'Color','white','FontSize',12)
            else
                text(j-dxScaled-0.1,i, accTxt,'Color','red','FontSize',12)
            end
        end
    end
end
