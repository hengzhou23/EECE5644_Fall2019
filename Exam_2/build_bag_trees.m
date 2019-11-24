function [bag_trees] = build_bag_trees(data, n)
    bag_trees = cell(n, 1);
    for i = 1:n
        % Generate new training data set
        random_data_index = randi([1, length(data)], 0.9*length(data), 1); % Get a 900x1 random index from data
        dataTrain = data(random_data_index,:);

        % Train a tree(same method to 1.b) and add it to the forest
        bag_trees{i} = fitctree(dataTrain(:,1:2), dataTrain(:,3), 'MaxNumSplits', 11, 'PredictorSelection', 'allsplits', ...
        'PruneCriterion', 'impurity', 'SplitCriterion', 'gdi');
    end
end