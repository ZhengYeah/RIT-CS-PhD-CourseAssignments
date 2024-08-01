for alpha = [0.05]
    if alpha == 0.05
        epsilon_0 = 1.95996;
    elseif alpha == 0.25
        epsilon_0 = 1.15034;
    end

    theta = [0 0.1 0.3 0.5 0.7 0.9];
    for k = 1:1:length(theta)
        for N = [10]
            normal_test_in = 0;
            ci_test_in = zeros(10, 1);

            mean_epsilon = 0;
            mean_a = zeros(10, 1);
            mean_b = zeros(10, 1);

            for i = 1:10000
                X = sample_bernoulli(N, theta(k));
                % Range estimation using CLT CI
                mean_X = mean(X);
                variance = std(X);
                epsilon = variance * epsilon_0 / sqrt(N);
                mean_epsilon = mean_epsilon + epsilon / 10000;

                if abs(mean_X - theta(k)) <= epsilon
                    normal_test_in = normal_test_in + 1;
                end
                % Range estimation using ci.m
                a = zeros(10, 1);
                b = zeros(10, 1);
                for j = 1:1:10
                    [a(j), b(j)] = ci(X, j);
                    if a(j) <= theta(k) && theta(k) <= b(j)
                        ci_test_in(j) = ci_test_in(j) + 1;
                    end
                end
                mean_a = mean_a + a / 10000;
                mean_b = mean_b + b / 10000;
            end

            fprintf("Bernoulli theta = %1.3f, mean epsilon: %1.3f, normal missed: %1.6f \n", theta(k), mean_epsilon, 1 - normal_test_in / 10000);
            for i = 1:1:10
                fprintf("Function %d: mean CI [%1.6f, %1.6f], missed: %1.6f \n", i, mean_a(i), mean_b(i), 1 - ci_test_in(i) / 10000);
            end
        end
    end
end