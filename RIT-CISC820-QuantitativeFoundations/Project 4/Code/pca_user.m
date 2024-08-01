function [principle_component, Y, mean_x] = pca_user(x, threshold, reconstruction)
% each column is an img
% N: number of image

[p, N] = size(x);
mean_x = mean(x, 2);
centered_x = x - repmat(mean_x, 1, N);

% SVD
[U, S, V] = svd(centered_x);
principle_component = U(:, 1:threshold);
% eigenface
% imshow(reshape(principle_component(:, 100), [112,92]), []);
Y = principle_component' * centered_x; 

if reconstruction
    % add center back
    compressed = Y' * principle_component' + repmat(mean_x, 1, N)';
    % reconstruction the first image
    compressed = reshape(compressed(1, :), [112,92]);
    % imshow(compressed, []);

    % distance from the original
    x_img = reshape(x(:, 1), [112,92]);
    fprintf("L2 distance: %1.3f", norm(x_img - compressed, 2));
end
