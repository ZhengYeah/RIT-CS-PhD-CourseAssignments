function accuracy = pca_classification()
train_data = load_dataset(true);
test_data = load_dataset(false);
train_data = train_data';
test_data = test_data';
[p, N] = size(test_data);

% use 200 eigenfaces
[principle_component, Y, mean_x] = pca_user(train_data, 200, false);
centered_test = test_data - repmat(mean_x, 1, N);
embedded_test = principle_component' * centered_test;
truth_test_label_1 = repelem([1:35]', 2);
truth_test_label =[truth_test_label_1; ones(50,1) * 36];
train_label = repelem([1:35]', 8);
one_hot_train = bsxfun(@eq, train_label(:), 1:35);

% linear representation: label = W * (principle_component' * train_data)
% pseudu-inverse
W = (one_hot_train' * Y') * pinv(Y * Y');
output =  W * embedded_test;


[score, idx] = max(output);
% accuracy = sum(idx == truth_test_label') / length(truth_test_label);
accuracy_1 = sum(idx(1:70) == truth_test_label(1:70)') / 70;
% fprintf("Total accuracy: %1.2f\n", accuracy);
fprintf("First-35 classes accuracy: %1.2f\n", accuracy_1);

end
