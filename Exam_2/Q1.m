close all; clear; clc;
T = readtable('Q1.csv');
data = table2array(T);
data_1 = T.Var1;
data_2 = T.Var2;
labels = T.Var3;

%% Q1.a 
% Provide a scatter plot of the entire data set, using the marker ¡®o¡¯ and color ¡®red¡¯ for class
% label -1 and marker ¡®x¡¯ and color ¡®black¡¯ for class label 1 with grid on
% and axies labelled
neg_class_rows = T.Var3==-1;
pos_class_rows = T.Var3==1;
X1 = data_1(neg_class_rows);
Y1 = data_2(neg_class_rows);
X2 = data_1(pos_class_rows);
Y2 = data_2(pos_class_rows);

figure;
scatter(X1, Y1, 'o', '.r');
hold on;
scatter(X2, Y2, 'x', '.k');
xlabel('Data 1');
ylabel('Data 2');
title('Scatter Plot of Data 1 (X) and Data 2 (Y)');
legend('Class -1', 'Class +1');
axis([-4 4 -4 4]);
grid on;
hold off;

%% Q1.b
% Split first 10 percent of data for testing and rest for training
dataTest = data(1:0.1*length(labels),:);
dataTrain = data((0.1*length(labels)+1):end,:);

% Train a decision tree model with split criterion as 'gdi' (Gini's diversity index)
mdl = fitctree(dataTrain(:,1:2), dataTrain(:,3), 'MaxNumSplits', 11, 'PredictorSelection', 'allsplits', ...
    'PruneCriterion', 'impurity', 'SplitCriterion', 'gdi');
view(mdl, 'Mode', 'Graph');

% Generate confusion matrix chart
predictedLabels = predict(mdl, dataTest(:, 1:2));
trueLabels = dataTest(:, 3);
figure;
cm = confusionchart(trueLabels, predictedLabels);
title('Confusion Matrix for the Decision Tree on the Test Data');

% Plot the decision boundaries for the decision tree
figure;
[x_grid, y_grid] = meshgrid((-4:.01:4), (-4:.01:4));
grid_matrix = [x_grid(:) y_grid(:)];
grid_labels = predict(mdl, grid_matrix);
gscatter(x_grid(:), y_grid(:), grid_labels, [0.5 0.5 1; 1 1 0.5]);
ax = gca;
ax.Layer = 'top';
hold on;
scatter(X1, Y1, 'o', '.r');
hold on;
scatter(X2, Y2, 'x', '.k');
xlabel('Data 1');
ylabel('Data 2');
title('Scatter Plot of Data 1 (X) and Data 2 (Y) with boundaries');
legend('Class -1 boundary', 'Class +1 boundary', 'Class -1', 'Class +1');
axis([-4 4 -4 4]);
grid on;
hold off;
% 
% % Determine the loss
% errTree = loss(mdl,dataTest);
% disp([num2str(errTree),' = Classification Tree Loss',])
% 
% % Prune the tree and recalculate the loss
% mdlPruned = prune(mdl,'Level',1);
% 
% % Determine the loss
% errTreePruned = loss(mdlPruned,dataTest);
% disp([num2str(errTreePruned),' = Pruned Classification Tree Loss',])

%% Q1.c
n = 7;
dataTest = data(1:0.1*length(labels),:);
[bag_forest] = build_bag_trees(data, n);

% Generate confusion matrix chart
predictedLabels_set = zeros(length(dataTest), n);
for i = 1:n
    predictedLabels_set(:, i) = predict(bag_forest{i}, dataTest(:, 1:2));
end
predictedLabels = mode(predictedLabels_set, 2); % Find the most-vote classification result
trueLabels = dataTest(:, 3);
figure;
cm = confusionchart(trueLabels, predictedLabels);
title('Confusion Matrix for the Bagging Decision Tree on the Test Data');

% Plot the decision boundaries for the decision tree
figure;
[x1_grid, x2_grid] = meshgrid((-4:.01:4), (-4:.01:4));
grid_matrix = [x1_grid(:) x2_grid(:)];
grid_label_set = zeros(length(grid_matrix), n);
for i = 1:n
    grid_label_set(:, i) = predict(bag_forest{i}, grid_matrix(:, 1:2));
end
grid_label = mode(grid_label_set, 2); % Find the most-vote classification result
gscatter(x1_grid(:), x2_grid(:), grid_label, [0.5 0.5 1; 1 1 0.5]);
ax = gca;
ax.Layer = 'top';
hold on;
scatter(X1, Y1, 'o', '.r');
hold on;
scatter(X2, Y2, 'x', '.k');
xlabel('Data 1');
ylabel('Data 2');
title('Scatter Plot of Data 1 and Data 2 with bagging boundaries');
legend('Class -1 boundary', 'Class +1 boundary', 'Class -1', 'Class +1');
axis([-4 4 -4 4]);
grid on;
hold off;

function [bag_forest] = build_bag_trees(data, n)
    bag_forest = cell(n, 1);
    for i = 1:n
        % Generate new training data
        random_data_index = randi(length(data), 1, 0.9*length(data)); 
        dataTrain2 = data(random_data_index,:);

        % Train a tree(same method to 1.b) and add it to the forest
        bag_forest{i} = fitctree(dataTrain2(:,1:2), dataTrain2(:,3), 'MaxNumSplits', 11, 'PredictorSelection', 'allsplits', ...
        'PruneCriterion', 'impurity', 'SplitCriterion', 'gdi');
    end
end

%% Q1.d
