function [gmm_data, gm_model] = gmm(data, cluster_num)
    fprintf("GMM with %d clusters\n", cluster_num)

    gm_model = fitgmdist(data, cluster_num,...
            'Replicates', 4,...
            'Options', statset('MaxIter',500,'TolFun',1e-6));

    class = gm_model.cluster(data);
    gmm_data = array2table([data class], 'VariableNames', {'X', 'Y', 'R', 'G', 'B', 'Class'});
end