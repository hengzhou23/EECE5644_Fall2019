function [bag_forest] = build_bag_forest(data, n)
    bag_forest = cell(n, 1);
    for i = 1:n
        % Generate new training data set
        random_data_index = randi(length(data), 1, 0.9*length(data)); 
        dataTrain = data(random_data_index,:);

        % Train a tree(same method to 1.b) and add it to the forest
        bag_forest{i} = fitctree(dataTrain(:,1:2), dataTrain(:,3), 'MaxNumSplits', 11, 'PredictorSelection', 'allsplits', ...
        'PruneCriterion', 'impurity', 'SplitCriterion', 'gdi');
    end
end