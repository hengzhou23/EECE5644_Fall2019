function [image, x_vals, y_vals] = image_generator(data, width, height)
    C = [[0.25,0.25,0.5];
         [0.9,0.1,0];
         [0.2,0.1,0.7];
         [0.4,0.2,0.4];
         [0,2,1];];
    image = zeros(img_height, img_width, 3);

    x_vals = rescale(data.X, 1, width);
    y_vals = rescale(data.Y, 1, height);

    len = length(data.X);

    for i = 1:len
        image(int16(y_vals(i)), int16(x_vals(i)), :) = C(int8(data.Class(i)), :);
    end
end