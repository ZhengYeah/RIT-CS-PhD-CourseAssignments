function [x_0, res, step_num, tmp_f] = newton_descent(f, f_gradient, f_hessin, tolerance, x_0, rho, c, fun_2_flag)

max_step_num = 1e3;
step_num = 0;
% used for ploting
tmp_f = [f(x_0)];

if fun_2_flag
    tmp_alpha = [];
end

while true
    g_0 = f_gradient(x_0);
    h_0 = f_hessin(x_0);
    p = h_0 \ (-g_0);
    alpha = backtracking_line_search(f, g_0, p, x_0, rho, c, fun_2_flag);
    if fun_2_flag
        tmp_alpha(end + 1) = alpha;
    end
    x = x_0 + p * alpha;
    % disp([f(x), f(x_0)]);
    tmp_f(end + 1) = f(x);


    if abs(f(x) - f(x_0)) < tolerance
        break;
    end
    step_num = step_num + 1;
    if step_num >= max_step_num
        break;
    end

    x_0 = x;
end

if fun_2_flag
    figure
    axis_x = 1:1:length(tmp_alpha);
    plot(axis_x, tmp_alpha, "-o");
end


res = f(x_0);
end
