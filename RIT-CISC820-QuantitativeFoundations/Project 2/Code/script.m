%%%%% test functions %%%%

% x_0 = zeros(100, 1);
% output_1 = fun_2(x_0);
% size(output_1)
% output_2 = fun_2_gradient(x_0);
% size(output_2)
% output_3 = fun_2_hessin(x_0);
% size(output_3)


tolerance = 1e-8;

% function 1
% x_0 = ones(100, 1);
% rho = 0.9;
% c = 0.1;
% % 
% % % [x_0, res, step_num, tmp_f] = gradient_descent(@fun_1, @fun_1_gradient, tolerance, x_0, rho, c, false);
% [x_0, res, step_num, tmp_f] = newton_descent(@fun_1, @fun_1_gradient, @fun_1_hessin, tolerance, x_0, rho, c, false);
% % [x_0, res, step_num, tmp_f] = quasi_newton(@fun_1, @fun_1_gradient, tolerance, x_0, rho, c, false);
% % 
% disp([res, step_num]);
% figure
% axis_x = 1:1:length(tmp_f);
% plot(axis_x, tmp_f, "-o");


% % function 2
% tolerance = 1e-5;
% 
% 
% x_0 = zeros(100, 1);
% rho = 0.9;
% c = 0.5;
% 
% % [x_0, res, step_num, tmp_f] = gradient_descent(@fun_2, @fun_2_gradient, tolerance, x_0, rho, c, true);
% [x_0, res, step_num, tmp_f] = newton_descent(@fun_2, @fun_2_gradient, @fun_2_hessin, tolerance, x_0, rho, c, true);
% % [x_0, res, step_num, tmp_f] = quasi_newton(@fun_2, @fun_2_gradient, tolerance, x_0, rho, c, true);
% % 
% disp([res, step_num]);
% figure
% axis_x = 1:1:length(tmp_f);
% plot(axis_x, tmp_f, "-o");



% % function 3
% x_0 = [100; 100];
% rho = 0.1;
% c = 0.9;
% 
% [x_0, res, step_num, tmp_f] = gradient_descent(@fun_3, @fun_3_gradient, tolerance, x_0, rho, c, false);
% % [x_0, res, step_num, tmp_f] = newton_descent(@fun_3, @fun_3_gradient, @fun_3_hessin, tolerance, x_0, rho, c, false);
% % [x_0, res, step_num, tmp_f] = quasi_newton(@fun_3, @fun_3_gradient, tolerance, x_0, rho, c, false);
% 
% figure
% axis_x = 1:1:length(tmp_f);
% plot(axis_x, tmp_f, "-o");
% disp([res, step_num]);
% disp(x_0)



% Adam Optimizer

% x_0 = ones(100, 1);
% [x_0, res, step_num, tmp_f] = adam_optimizer(@fun_1, @fun_1_gradient, x_0, 3e-4, 1e-8, 0.9, 0.99, tolerance, false);
% figure
% axis_x = 1:1:length(tmp_f);
% plot(axis_x, tmp_f, "-o");
% disp([res, step_num]);
% disp(x_0)


x_0 = zeros(100, 1);
[x_0, res, step_num, tmp_f] = adam_optimizer(@fun_2, @fun_2_gradient, x_0, 1e-3, 1e-8, 0.9, 0.99, tolerance, true);
figure
axis_x = 1:1:length(tmp_f);
plot(axis_x, tmp_f, "-o");
disp([res, step_num]);
disp(x_0)


% 
% x_0 = [100; 100];
% [x_0, res, step_num, tmp_f] = adam_optimizer(@fun_3, @fun_3_gradient, x_0, 3e-4, 1e-8, 0.9, 0.99, tolerance, false);
% 
% figure
% axis_x = 1:1:length(tmp_f);
% plot(axis_x, tmp_f, "-o");
% disp([res, step_num]);
% disp(x_0)



