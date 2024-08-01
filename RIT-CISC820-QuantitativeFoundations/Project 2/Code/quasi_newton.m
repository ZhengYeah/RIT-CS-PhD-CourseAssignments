function [x_0, res, step_num, tmp_f] = quasi_newton(f, f_gradient, tolerance, x_0, rho, c, fun_2_flag)

max_step_num = 1e3;
step_num = 0;
F = eye(length(x_0));

% used for ploting
tmp_f = [f(x_0)];


while true
    g_0 = f_gradient(x_0);
    p = -F * g_0;
    alpha = backtracking_line_search(f, g_0, p, x_0, rho, c, fun_2_flag);
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

    % DFP
    s = x - x_0;
    y = f_gradient(x) - g_0;
    F = F + ((y' * (F * y + s)) / (y' * s)^2) * (s*s') - (s * y' * F + F * y * s') / (y' * s);
    x_0 = x;
end

res = f(x_0);
end
