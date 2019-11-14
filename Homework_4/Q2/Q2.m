close all; clear all; clc;

N = 1000;
n = 2;
K = 10;
p = [0.35, 0.65]; % Class priors [q-, q+]
x = zeros(n, N);

% (1) Generate Training Data

% Class -1 is labelled with 0
% Class +1 is labelled with 1
label = rand(1,N) >= p(1);  % Generate class samples randomly
n_samples = [length(find(label==0)), length(find(label==1))];  % Number of samples in each class
 
% Generate data for class -1
mu = [0;0];
sigma = [1 0;
         0 1];
temp = mvnrnd(mu, sigma, n_samples(1));
x(:, label==0) = transpose(temp);

% Generate data for class +1 
% Multiply the angle random number by 2pi and shift left by pi since its interval is [-pi, +pi]
% Shift the radius random number by 2 since its interval is [2, 3]
angle = (2*pi) * rand(1, n_samples(2)) - pi;
radius = rand(1, n_samples(2)) + 2;
data(1,:) = radius .* sin(angle);
data(2,:) = radius .* cos(angle);
x(:, label==1) = data;

% Plot the generated data
class_one = x(:,label==0);
class_two = x(:,label==1);
figure();
plot(class_one(1,:), class_one(2,:), '.r')
hold on;
plot(class_two(1,:), class_two(2,:), '.b')
axis equal
title('Training Data Generated')
xlabel('Feature 1')
ylabel('Feature 2')
legend('class -1', 'class +1')
hold off;
