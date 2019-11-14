function [image, x_vals, y_vals] = image_generator(data_T, img_width, img_height)
    color_list = [[0.25,0.25,0.5];
                  [1,0,0];
                  [0.2,0.1,0.7];
                  [1,0,1];
                  [0,1,1];];
    colors  = {'.b','.k','.r','.g','.y'};
    image = zeros(img_height, img_width, 3);

    x_vals = rescale(data_T.X, 1, img_width);
    y_vals = rescale(data_T.Y, 1, img_height);
%     y_vals = linspace(0, 1, img_height);
%     x_vals = linspace(0, 1, img_width);
    
    T_len = length(data_T.X);

    for i = 1:T_len
        image(int16(y_vals(i)), int16(x_vals(i)), :) = color_list(int8(data_T.Class(i)), :);
    end
end