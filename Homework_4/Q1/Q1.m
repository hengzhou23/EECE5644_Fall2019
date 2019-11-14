close all;clc;

% Load image data in each image 
% and get the sets of 5-dimensional normalized feature vectors
bird_raw_data = imread('42049_colorBird.jpg');
[r, g, b, normalized_bird_data] = image_loader(bird_raw_data);

plane_raw_data = imread('3096_colorPlane.jpg');
[r, g, b, normalized_plane_data] = image_loader(plane_raw_data);

% K values: [2 3 4 5]
for K = 2:5
    image_classifier(bird_raw_data, normalized_bird_data, K);
    image_classifier(plane_raw_data, normalized_plane_data, K);
end


function [] = image_classifier(raw_data, normalized_data, K)
    % K-means and GMM classification
    [kmeans_data, C, sumD, D] = k_means(normalized_data, K);
    [label, Centers] = imsegkmeans(raw_data, K); % Assign labels to each pixel
    [gmm_data, gm_mdl] = gmm(normalized_data, K); % Repeat the segmentation with GMM

    % Generate images 
    kmean_img = image_generator(kmeans_data, 481, 321);
    seg_img = labeloverlay(raw_data, label); % Overlay the original image with labeled matrices
    gmm_img = image_generator(gmm_data, 481, 321);

    % Show the images
    figure;
    imshow(kmean_img);
    title(['K-Means with ', int2str(K), ' clusters'])
    figure;
    imshow(gmm_img);
    title(['GMM with ', int2str(K), ' clusters'])
    figure;
    imshow(seg_img);
    title(['Segment the image with K = ', int2str(K), ' according to label values'])
end
