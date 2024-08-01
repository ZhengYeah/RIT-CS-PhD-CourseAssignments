function accuracy = pca_recognization(score_threshod)
train_data = load_dataset(true);
test_data = load_dataset(false);
train_data = train_data';
test_data = test_data';
[p, N] = size(test_data);

% other img
test_other = zeros(1, 92 * 112);
img_read = imread("./butt.pgm");
img_read = img_read(:,:,1);
% imshow(img_read);

test_other(1,:) = reshape(img_read, 1, []);
test_other = test_other';

% use 200 eigenfaces
[principle_component, Y, mean_x] = pca_user(train_data, 200, false);
centered_test = test_data - repmat(mean_x, 1, N);
embedded_test = principle_component' * centered_test;
centered_test_other = test_other - repmat(mean_x, 1, N);
embedded_test_other = principle_component' * test_other;

truth_test_label_1 = repelem([1:35]', 2);
truth_test_label =[truth_test_label_1; ones(50,1) * 36];
train_label = repelem([1:35]', 8);
one_hot_train = bsxfun(@eq, train_label(:), 1:35);

% linear representation: label = W * (principle_component' * train_data)
% pseudu-inverse
W = (one_hot_train' * Y') * pinv(Y * Y');

% test face img
output =  W * embedded_test;
[score, idx] = max(output);
average_score = mean(score);

% other img
output_other = W * embedded_test_other;
score = max(output_other);

end
