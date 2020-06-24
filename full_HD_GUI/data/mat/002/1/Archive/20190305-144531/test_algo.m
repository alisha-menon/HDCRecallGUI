close all
clear
clc

load('002_1_201_20190305-144531_1.mat');
rest = double(data(5001:5500,1:64));
rest(rest>2^15) = rest(rest>2^15) - 2^15;

f = hw_features(rest,50,32);