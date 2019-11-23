close all; clear all; clc;

dataTrain = readtable('Q2train.csv');
dataTrain = table2array(dataTrain); 
dataTest = readtable('Q2test.csv');
dataTest = table2array(dataTest);

%% Q2.2
figure;
plot(dataTrain(:,2),dataTrain(:,3), 'o:r')
xlabel('Longitude');
ylabel('Latitude');
title('Scatter Plot of the Measurements in Two Dimensions');
grid on;