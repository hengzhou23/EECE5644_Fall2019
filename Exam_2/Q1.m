close all force; clear; clc;
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
initial_weight = ones(900, 1) / 900; % initial weight
population = 1:900;
w = initial_weight;
adaboost_trees = cell(n,2);

for i = 1:level
    % Fit a tree classifier with weight
    dataIdx = randsample(population, 900, true, w);
    dataTrain = X(dataIdx(:, 1:900), :);
    temp_mdl = fitctree(dataTrain(:,1:2), dataTrain(:,3), 'MaxNumSplits', 11, 'PredictorSelection', 'allsplits', ...
    'PruneCriterion', 'impurity', 'SplitCriterion', 'gdi');
    adaboost_trees{i,1} = temp_mdl; 
    
    % Compute the error
    tempLabel = predict(temp_mdl, dataTrain(:,1:2));
    inequality = (tempLabel ~= dataTrain(:,3));
    err = sum(w .* inequality ) / sum(w);
    
    % Compute the level weight
    a = log((1 - err) / err);
    adaboost_trees{i,2} = a;
    
    % Update weights
    w = w .* exp(a .* inequality);    % compute un-normalized weights
    w = w / sum(w);                 % renormalize the weights
end
final_weight = w;
initial_final_W = [initial_weight, final_weight];
level_weights = [adaboost_trees{:,2}] / sum([adaboost_trees{:,2}])

% Compute Output G(x) of each label in test data set
label_set = zeros(length(dataTest), i);
for i = 1:level
    alpha = level_weights(i);
    label_set(:, i) = alpha .* predict(adaboost_trees{i,1}, dataTest(:,1:2));
end
Gx = sign(sum(transpose(label_set)));
labels = transpose(Gx);
% Generate confusion matrix chart
trueLabels = dataTest(:, 3);
figure;
confusionchart(trueLabels, labels);
title('Confusion Matrix for the Boosting Decision Tree on the Test Data');

% Plot the decision boundaries for the decision tree
figure;
[Xg, Yg] = meshgrid((-4:.01:4), (-4:.01:4));
Xgrid = [Xg(:) Yg(:)];
grid_label_set = zeros(length(Xgrid), i);
% Compute Output G(x) of each label
for i = 1:level
    alpha = level_weights(i);
    grid_label_set(:, i) = alpha .* predict(adaboost_trees{i,1}, Xgrid(:, 1:2));
end
Gx = sign(sum(transpose(grid_label_set)));
grid_labels = transpose(Gx);
gscatter(Xg(:), Yg(:), grid_labels, [0.5 0.5 1; 1 1 0.5]);
ax = gca;
ax.Layer = 'top';
hold on;
plot_scatter(X);
title('Scatter Plot with Boundaries by Classification with Boosting');
legend('Class -1 boundary', 'Class +1 boundary', 'Class -1', 'Class +1');
hold off;


