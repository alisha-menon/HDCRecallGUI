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
    prediction = double(data(:,63));
    distance = double(data(:,64))';
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