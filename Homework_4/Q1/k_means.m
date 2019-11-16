function [kmeans_data, C, sumD, D] = k_means(data, cluster_num)
    [idx,C,sumD,D] = kmeans(data, cluster_num, 'Distance','sqeuclidean', 'Replicates', 5, 'MaxIter', 500, 'Start', 'cluster');
    kmeans_data = array2table([data idx], 'VariableNames', {'X', 'Y', 'R', 'G', 'B', 'Class'});
end    