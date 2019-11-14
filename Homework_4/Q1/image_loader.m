function [r, g, b, n_row, n_col, norm_image_data] = image_loader(image_data)
    % Get number of row pixels and column pixels and feature size
    row = linspace(0, 1, size(image_data,1));
    col = linspace(0, 1, size(image_data,2));
    [n_row, n_col, ~] = size(image_data)
    
    % Normalize red, green, and blue values of the image color at each pixel
    % Then linearly shifting and scaling the values to the interval [0,1]
    r = rescale(image_data(:,:,1));
    g = rescale(image_data(:,:,2));
    b = rescale(image_data(:,:,3));
    
    % Insert values to the set of 5-dimensional normalized feature vectors
    norm_image_data = zeros(n_row.*n_col, 5);
    k = 1;
    for i = 1:n_row
        for j = 1:n_col
            norm_image_data(k,1) = col(j);
            norm_image_data(k,2) = row(i);
            norm_image_data(k,3) = r(i,j);
            norm_image_data(k,4) = g(i,j);
            norm_image_data(k,5) = b(i,j);
            k = k + 1;
        end
    end
end

