function res = fun_2_hessin(x)
% x: [100, 1] vector

fid = fopen('fun2_A.txt','r');
  A = fscanf(fid,'%e ',[500, 100]);
  fclose(fid);
fid = fopen('fun2_b.txt','r');
  b = fscanf(fid,'%e ',[500, 1]);
  fclose(fid);
fid = fopen('fun2_c.txt','r');
  c = fscanf(fid,'%e ',[100, 1]);
  fclose(fid);

res = A' * diag(1 ./ (b - A * x).^2) * A;
end

