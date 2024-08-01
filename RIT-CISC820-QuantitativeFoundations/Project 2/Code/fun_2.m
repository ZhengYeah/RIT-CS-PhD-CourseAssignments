function res = fun_2(x)
% x: [100, 1] vector

fid = fopen('fun2_A.txt','r');
A = fscanf(fid, '%e', [500, 100]);
fclose(fid);
fid = fopen('fun2_b.txt','r');
b = fscanf(fid,'%e', [500, 1]);
fclose(fid);
fid = fopen('fun2_c.txt','r');
c = fscanf(fid,'%e', [100, 1]);
fclose(fid);

% = sum log b - A'X
% tmp = log(b - A * x);
% disp(tmp)

res = c' * x - sum(log(b - A * x));
end

