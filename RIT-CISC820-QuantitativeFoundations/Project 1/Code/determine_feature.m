function determine_feature()
train_file = "traindata.txt";

traindata = importdata(train_file);
X = traindata(1:900,1:8);
Y = traindata(1:900,9);


% % feature matrix
% % p(x1,x2,...x8)

% found degree: 3   2	1	3	3	1	1	3 with mean batch error 58.4311881781144

max_deg = 3;
for deg_1 = 1:max_deg
    for deg_2 = 1:max_deg
        for deg_3 = 1:max_deg
            for deg_4 = 1:max_deg
                for deg_5 = 1:max_deg
                    for deg_6 = 1:max_deg
                        for deg_7 = 1:max_deg
                            for deg_8 = 1:max_deg

                                % % training and cross validation
                                K = 9;
                                N = size(X,1);
                                error_list = zeros(K,1);

                                for k = 1:K
                                    test_X = X((k-1)*N/K+1:k*N/K,:);
                                    test_Y = Y((k-1)*N/K+1:k*N/K);
                                    training_X = [X(1:(k-1)*N/K,:); X(k*N/K:N,:)];
                                    training_Y = [Y(1:(k-1)*N/K); Y(k*N/K:N)];

                                    feature_X = [X_degree(training_X(:,1),deg_1) X_degree(training_X(:,2),deg_2) X_degree(training_X(:,3),deg_3) ... 
                                        X_degree(training_X(:,4),deg_4) X_degree(training_X(:,5),deg_5) X_degree(training_X(:,6),deg_6) ...
                                        X_degree(training_X(:,7),deg_7) X_degree(training_X(:,8),deg_8) ones(size(training_X,1),1)];
                                    p = feature_X \ training_Y;

                                    test_feature_X = [X_degree(test_X(:,1),deg_1) X_degree(test_X(:,2),deg_2) X_degree(test_X(:,3),deg_3) ...
                                        X_degree(test_X(:,4),deg_4) X_degree(test_X(:,5),deg_5) X_degree(test_X(:,6),deg_6) ...
                                        X_degree(test_X(:,7),deg_7) X_degree(test_X(:,8),deg_8) ones(size(test_X,1),1)];
                                    pred_Y = test_feature_X * p;

                                    error_list(k) = norm(test_Y - pred_Y)^2 / size(test_X,1);                                    
                                end
                                order_now = [deg_1,deg_2,deg_3,deg_4,deg_5,deg_6,deg_7,deg_8,mean(error_list)];
                                writematrix(order_now,"mean_error_data.csv", "WriteMode","append");
                            end
                        end
                    end
                end
            end
        end
    end
end
end


function res = X_degree(X, deg)
res = [];
while deg > 0
    res =[res X.^deg];
    deg = deg - 1;
end
end














