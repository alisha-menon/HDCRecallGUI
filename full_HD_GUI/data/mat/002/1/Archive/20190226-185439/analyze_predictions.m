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

plot(correct_all)
hold on
plot(predict_all)

accuracy = sum(correct_all == predict_all)/length(predict_all)