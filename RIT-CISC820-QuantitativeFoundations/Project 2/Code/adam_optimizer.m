function [x_0, res, step_num, tmp_f] = adam_optimizer(f, f_gradient, x_0, alpha, eps, beta_1, beta_2, tolerance, fun_2_flag)
    
max_step_num = 1e4;
step_num = 0;

update = 1;

% used for ploting
tmp_f = [f(x_0)];

while true

    if fun_2_flag
        fid = fopen('fun2_A.txt','r');
        A = fscanf(fid, '%e', [500, 100]);
        fclose(fid);
        fid = fopen('fun2_b.txt','r');
        b = fscanf(fid,'%e', [500, 1]);

        while ~all(b > A * (x_0 - update))
            update = adam_optimizer_step(f_gradient, x_0, alpha, eps, beta_1, beta_2);
        end
    else
        update = adam_optimizer_step(f_gradient, x_0, alpha, eps, beta_1, beta_2);
    end
    x = x_0 - update;
    disp("f(x), f(x_0) = ");
    disp([f(x), f(x_0)]);
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
res = f(x_0);
end
