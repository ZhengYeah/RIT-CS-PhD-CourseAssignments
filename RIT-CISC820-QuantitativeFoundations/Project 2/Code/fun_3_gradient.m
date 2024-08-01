function res = fun_3_gradient(x)
% row vector
res = [-400 * (x(2) - x(1)^2) * x(1) - 2 * (1 - x(1)); 200 * (x(2) - x(1)^2)];  
end

