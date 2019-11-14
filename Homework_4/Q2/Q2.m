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

l = 2*(label-0.5);
dummy = ceil(linspace(0,N,K+1));
    for k = 1:K, indPartitionLimits(k,:) = [dummy(k)+1,dummy(k+1)]; end,
    CList = 10.^linspace(-3,7,11); 
    for CCounter = 1:length(CList)
        [CCounter,length(CList)],
        C = CList(CCounter);
        for k = 1:K
            indValidate = [indPartitionLimits(k,1):indPartitionLimits(k,2)];
            xValidate = x(:,indValidate); % Using folk k as validation set
            lValidate = l(indValidate);
            if k == 1
                indTrain = [indPartitionLimits(k,2)+1:N];
            elseif k == K
                indTrain = [1:indPartitionLimits(k,1)-1];
            else
                indTrain = [indPartitionLimits(k-1,2)+1:indPartitionLimits(k+1,1)-1];
            end
            % using all other folds as training set
            xTrain = x(:,indTrain); lTrain = l(indTrain);
            SVMk = fitcsvm(xTrain',lTrain,'BoxConstraint',C,'KernelFunction','linear');
            dValidate = SVMk.predict(xValidate')'; % Labels of validation data using the trained SVM
            indCORRECT = find(lValidate.*dValidate == 1); 
            Ncorrect(k)=length(indCORRECT);
        end 
        PCorrect(CCounter)= sum(Ncorrect)/N; 
    end 
    figure(1), subplot(1,2,1),
    plot(log10(CList),PCorrect,'.',log10(CList),PCorrect,'-'),
    xlabel('log_{10} C'),ylabel('K-fold Validation Accuracy Estimate'),
    title('Linear-SVM Cross-Val Accuracy Estimate'), %axis equal,
    [dummy,indi] = max(PCorrect(:)); [indBestC, indBestSigma] = ind2sub(size(PCorrect),indi);
    CBest= CList(indBestC); 
    SVMBest = fitcsvm(x',l','BoxConstraint',CBest,'KernelFunction','linear');
    d = SVMBest.predict(x')'; % Labels of training data using the trained SVM
    indINCORRECT = find(l.*d == -1); % Find training samples that are incorrectly classified by the trained SVM
    indCORRECT = find(l.*d == 1); % Find training samples that are correctly classified by the trained SVM
    figure(1), subplot(1,2,2), 
    plot(x(1,indCORRECT),x(2,indCORRECT),'g.'), hold on,
    plot(x(1,indINCORRECT),x(2,indINCORRECT),'r.'), axis equal,
    pTrainingError = length(indINCORRECT)/N, % Empirical estimate of training error probability
    plotTitle = strcat('Training Data Classified by Linear SVM (RED: Incorrectly Classified) Pe=', num2str(pTrainingError*100));
    title(plotTitle),
    Nx = 1001; Ny = 990; xGrid = linspace(-10,10,Nx); yGrid = linspace(-10,10,Ny);
    [h,v] = meshgrid(xGrid,yGrid); dGrid = SVMBest.predict([h(:),v(:)]); zGrid = reshape(dGrid,Ny,Nx);
    figure(1), subplot(1,2,2), contour(xGrid,yGrid,zGrid,0); xlabel('x1'), ylabel('x2'), axis equal,
    
    % Display attributes for fitcsvm parameters
    disp(CBest);
    
    % Store Best Linear SVM Classifier
    linear_svm_best = SVMBest;
