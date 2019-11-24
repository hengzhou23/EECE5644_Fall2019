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
bag_forest = build_bag_forest(X, n); % Generate multiple trees

% Generate confusion matrix chart
predictedLabels_set = zeros(length(dataTest), n);
for i = 1:n
    predictedLabels_set(:, i) = predict(bag_forest{i}, dataTest(:, 1:2));
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
    grid_label_set(:, i) = predict(bag_forest{i}, Xgrid(:, 1:2));
end
predicted_species = mode(grid_label_set, 2); % Find the most-vote classification result
gscatter(Xg(:), Yg(:), predicted_species, [0.5 0.5 1; 1 1 0.5]);
set(gca, 'Layer', 'top')
hold on;
plot_scatter(X);
title('Scatter Plot with Boundaries by Classification with bagged trees');
legend('Class -1 boundary', 'Class +1 boundary', 'Class -1', 'Class +1');
hold off;

%% Q1.d
n = 7;
dataTest = X(1:0.1*length(labels),:);
len  = size(X,1);
w    = ones(len, 1) / len;
pop  = 1:len;
mdls = cell(n,2); 
ifW  = [w,zeros(len, 1)];
dataIdx = randsample(pop, len, true, w);
dataTrain = X(dataIdx, :);

for i = 1:n
    % Train the tree
    temp_mdl = fitctree(dataTrain(:,1:2), dataTrain(:,3), 'MaxNumSplits', 11, 'PredictorSelection', 'allsplits', ...
    'PruneCriterion', 'impurity', 'SplitCriterion', 'gdi');
    mdls{i,1} = temp_mdl; 
    
    % Get predicted labels
    tempLabel = predict(temp_mdl, dataTrain(:,1:2));
    trueness = (tempLabel ~= dataTrain(:,2));
    
    % Calculate the error
    err = sum(w .* trueness) / sum(w);
    
    % Calculate the classifier weight
    a = log((1 - err) / err);
    mdls{i,2} = log((1 - err) / err);  % save the weight of the model
    
    % Update weights
    w = w .* exp(a .* trueness);    % find un-normalized weights
    w = w / sum(w);                 % normalize weights
    if any(w < 0)
        w(w<0)
    end
    if i == n
        ifW(:,2) = w;
    end
end

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
    predictedLabels_set(:, i) = predict(bag_forest{i}, dataTest(:, 1:2));
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
    grid_label_set(:, i) = predict(bag_forest{i}, Xgrid(:, 1:2));
end
grid_label = mode(grid_label_set, 2); % Find the most-vote classification result
gscatter(Xg(:), Yg(:), grid_label, [0.5 0.5 1; 1 1 0.5]);
ax = gca;
ax.Layer = 'top';
hold on;
scatter(X1, Y1, 'o', '.r');
hold on;
scatter(X2, Y2, 'x', '.k');
xlabel('Data 1');
ylabel('Data 2');
title('Scatter Plot of Data 1 and Data 2 with boosting boundaries');
legend('Class -1 boundary', 'Class +1 boundary', 'Class -1', 'Class +1');
axis([-4 4 -4 4]);
grid on;
hold off;


