function alpha = backtracking_line_search(f, g_0, p, x_0, rho, c, fun_2_flag)
    % p: direction
    % c: slope

    alpha = 1;
    f_0 = f(x_0);
    
    if fun_2_flag
        fid = fopen('fun2_A.txt','r');
        A = fscanf(fid, '%e', [500, 100]);
        fclose(fid);
        fid = fopen('fun2_b.txt','r');
        b = fscanf(fid,'%e', [500, 1]);

        while ~all(b > A * (x_0 + alpha * p) ) || f(x_0 + alpha * p) > f_0 + c * alpha * (dot(p, g_0))
            alpha = rho * alpha;
        end
    else
        x = x_0 + alpha * p;
        f_next = f(x);

        while f_next > f_0 + c * alpha * (dot(p, g_0))
            alpha = rho * alpha;
            x = x_0 + alpha * p;
            f_next = f(x);
        end
    end
end

