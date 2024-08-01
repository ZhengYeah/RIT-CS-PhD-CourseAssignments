function pred_Y = final_model()
    train_file = "traindata.txt";
    test_file = "testinputs.txt";

    traindata = importdata(train_file);
    X = traindata(:,1:8);
    Y = traindata(:,9);
    testdata = importdata(test_file);
    X_test = testdata(:,:);

    % feature
    % use p(X) = [^3 ^2 ^1 ^3 ^3 ^1 ^1 ^3, 1]
    feature_X = [X_degree(X(:,1),3) X_degree(X(:,2),2) X_degree(X(:,3),1) ... 
        X_degree(X(:,4),3) X_degree(X(:,5),3) X_degree(X(:,6),1) ... 
        X_degree(X(:,7),1) X_degree(X(:,8),3) ones(size(X,1),1)];
    p = feature_X \ Y;

    pred_Y = feature_X * p;
    mean_error = norm(Y - pred_Y)^2 / size(X,1);
    disp(mean_error)

    % Plot true values (Y_true) as Yellow Circle
    plot(Y, 'yo', 'MarkerSize', 8, 'MarkerFaceColor', 'y', 'LineWidth', 1.5);
    hold on;

    % Plot predicted values (Y_pred) as red diamonds
    plot(pred_Y, 'rd', 'MarkerSize', 8, 'MarkerFaceColor', 'r', 'LineWidth', 1.5);

    % Customize the title and add a legend
    title('Comparison of Y True and Y Predicted');
    legend('Y True', 'Y Predicted', 'Location', 'best');
    grid on;
    % Customize the appearance of the plot
    set(gca, 'FontSize', 14);


    feature_X_test = [X_degree(X_test(:,1),3) X_degree(X_test(:,2),2) X_degree(X_test(:,3),1) ... 
        X_degree(X_test(:,4),3) X_degree(X_test(:,5),3) X_degree(X_test(:,6),1) ... 
        X_degree(X_test(:,7),1) X_degree(X_test(:,8),3) ones(size(X_test,1),1)];
    pred_Y = feature_X_test * p;
end


function res = X_degree(X, deg)
res = [];
while deg > 0
    res =[res X.^deg];
    deg = deg - 1;
end
end


    