for alpha = [0.05]
    if alpha == 0.05
        epsilon_0 = 1.95996;
    elseif alpha == 0.25
        epsilon_0 = 1.15034;
    end

    left = [0 0 0.2 0.4 0.6 0.7 0.9];
    right = [1 0 0.3 0.5 0.65 0.72 0.95];
    mu = (left + right) / 2;

    for k = 1:1:length(left)
        for N = [10]
            normal_test_in = 0;
            ci_test_in = zeros(10, 1);

            mean_epsilon = 0;
            mean_a = zeros(10, 1);
            mean_b = zeros(10, 1);

            for i = 1:10000
                X = sample_uniform(N, left(k), right(k));

                mean_X = mean(X);
                sigma = std(X);
                epsilon = sigma * epsilon_0 / sqrt(N);
                mean_epsilon = mean_epsilon + epsilon / 10000;
                if abs(mean_X - mu(k)) <= epsilon
                    normal_test_in = normal_test_in + 1;
                end

                a = zeros(10, 1);
                b = zeros(10, 1);
                for j = 1:1:10
                    [a(j), b(j)] = ci(X, j);
                    if a(j) <= mu(k) && mu(k) <= b(j)
                        ci_test_in(j) = ci_test_in(j) + 1;
                    end
                end
                mean_a = mean_a + a / 10000;
                mean_b = mean_b + b / 10000;
            end

            fprintf("Uniform [%1.3f, %1.3f], mean CI [mean]epsilon: %1.3f, normal missed: %1.6f \n", left(k), right(k), mean_epsilon, 1 - normal_test_in / 10000);
            for i = 1:1:10
                fprintf("Function %d: mean CI [%1.3f, %1.3f], missed: %1.6f \n", i, mean_a(i), mean_b(i), 1 - ci_test_in(i) / 10000);
            end
        end
    end
end