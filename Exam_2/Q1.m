close all; clear; clc;
T = readtable('Q1.csv');
X = table2array(T);
labels = X(:,3);

%% Q1.a 
% Provide a scatter plot of the entire data set, using the marker ¡®o¡¯ and color ¡®red¡¯ for class
% label -1 and marker ¡®x¡¯ and color ¡®black¡¯ for class label 1 with grid on
% and axies labelled
figure;
gscatter(X(:,1), X(:,2), labels, 'rk', 'ox');
xlabel('Data 1');
ylabel('Data 2');
title('Scatter Plot of Data 1 (X) and Data 2 (Y)');
legend('Class -1', 'Class +1');
axis([-4 4 -4 4]);
grid on;


%% Q1.b
% Split first 10 percent of data for testing and rest for training
dataTest = X(1:0.1*length(labels),:);
dataTrain = X((0.1*length(labels)+1):end,:);

% Train a decision tree model with split criterion as 'gdi' (Gini's diversity index)
mdl = fitctree(dataTrain(:,1:2), dataTrain(:,3), 'MaxNumSplits', 11, 'PredictorSelection', 'allsplits', ...
    'PruneCriterion', 'impurity', 'SplitCriterion', 'gdi');
view(mdl, 'Mode', 'Graph');

% Generate confusion matrix chart
predictedLabels = predict(mdl, dataTest(:, 1:2));
trueLabels = dataTest(:, 3);
figure;
confusionchart(trueLabels, predictedLabels);
title('Confusion Matrix for the Decision Tree on the Test Data');

% Plot the decision boundaries for the decision tree
figure;
[Xg, Yg] = meshgrid((-4:.01:4), (-4:.01:4));
Xgrid = [Xg(:) Yg(:)];
predicted_species = predict(mdl, Xgrid);
gscatter(Xg(:), Yg(:), predicted_species, [0.5 0.5 1; 1 1 0.5]);
set(gca, 'Layer', 'top')
hold on;
plot_scatter(X);
title('Scatter Plot with Boundaries by Classification in ID3');
legend('Class -1 boundary', 'Class +1 boundary', 'Class -1', 'Class +1');
hold off;


%% Q1.c
n = 7;
dataTest = X(1:0.1*length(labels),:);
bag_trees = build_bag_trees(X, n); % Generate multiple trees

% Generate confusion matrix chart
predictedLabels_set = zeros(length(dataTest), n);
for i = 1:n
    predictedLabels_set(:, i) = predict(bag_trees{i}, dataTest(:, 1:2));
end
predictedLabels = mode(predictedLabels_set, 2); % Find the most-vote classification result
trueLabels = dataTest(:, 3);
figure;
confusionchart(trueLabels, predictedLabels);
title('Confusion Matrix for the Bagging Decision Tree on the Test Data');

% Plot the decision boundaries for the decision tree
figure;
[Xg, Yg] = meshgrid((-4:.01:4), (-4:.01:4));
Xgrid = [Xg(:) Yg(:)];
grid_label_set = zeros(length(Xgrid), i);
for i = 1:i
    grid_label_set(:, i) = predict(bag_trees{i}, Xgrid(:, 1:2));
end
predicted_species = mode(grid_label_set, 2); % Find the most-vote classification result
gscatter(Xg(:), Yg(:), predicted_species, [0.5 0.5 1; 1 1 0.5]);
set(gca, 'Layer', 'top')
hold on;
plot_scatter(X);
title('Scatter Plot with Boundaries by Classification with Bagging');
legend('Class -1 boundary', 'Class +1 boundary', 'Class -1', 'Class +1');
hold off;

%% Q1.d
level = 7;
dataTest = X(1:0.1*length(labels),:);
N = length(X(:,1));
initial_weight = ones(N, 1) / N; % initial weight
initial_weight = initial_weight(1:900);
population = 1:900;
w = initial_weight(1:900);
mdls = cell(n,2);
dataIdx = randsample(population, 900, true, w);
dataTrain = X(dataIdx(:, 1:900), :);
for i = 1:level
    % Fit a tree classifier with weight
    temp_mdl = fitctree(dataTrain(:,1:2), dataTrain(:,3), 'MaxNumSplits', 11, 'PredictorSelection', 'allsplits', ...
    'PruneCriterion', 'impurity', 'SplitCriterion', 'gdi');
    mdls{i,1} = temp_mdl; 
    
    % Compute the error
    tempLabel = predict(temp_mdl, dataTrain(:,1:2));
    inequality = (tempLabel ~= dataTrain(:,2));
    err = sum(w .* inequality ) / sum(w);
    
    % Compute the classifier weight
    a = log((1 - err) / err);
    mdls{i,2} = a;
    
    % Update weights
    w = w .* exp(a .* inequality);    % compute un-normalized weights
    w = w / sum(w);                 % renormalize the weights
end
final_weight = w;
initial_final_W = [initial_weight, final_weight];

% Predict labels
X = dataTest(:,1:2);
allLabels = zeros(length(X), length(mdls));
for m = 1:size(mdls,1)
    a = mdls{m,2};
    allLabels = a .* predict(mdls{m,1}, X);
end
labels = sum(allLabels,2);
labels = labels ./ labels(1);

% Generate confusion matrix chart
predictedLabels_set = zeros(length(dataTest), n);
for i = 1:n
    predictedLabels_set(:, i) = predict(bag_trees{i}, dataTest(:, 1:2));
end
predictedLabels = mode(predictedLabels_set, 2); % Find the most-vote classification result
trueLabels = dataTest(:, 3);
figure;
cm = confusionchart(trueLabels, predictedLabels);
title('Confusion Matrix for the Boosting Decision Tree on the Test Data');

% Plot the decision boundaries for the decision tree
figure;
[Xg, Yg] = meshgrid((-4:.01:4), (-4:.01:4));
Xgrid = [Xg(:) Yg(:)];
grid_label_set = zeros(length(Xgrid), i);
for i = 1:i
    grid_label_set(:, i) = predict(bag_trees{i}, Xgrid(:, 1:2));
end
grid_label = mode(grid_label_set, 2); % Find the most-vote classification result
gscatter(Xg(:), Yg(:), grid_label, [0.5 0.5 1; 1 1 0.5]);
ax = gca;
ax.Layer = 'top';
hold on;
plot_scatter(X);
title('Scatter Plot with Boundaries by Classification with Boosting');
legend('Class -1 boundary', 'Class +1 boundary', 'Class -1', 'Class +1');
hold off;


