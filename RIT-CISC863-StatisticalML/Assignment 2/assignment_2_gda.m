meas_0 = [67, 79, 71]; 
meas_1 = [68, 67, 60];
species = [0, 1];
N_0 = length(meas_0);
N_1 = length(meas_1);


mu_0 = mean(meas_0);
mu_1 = mean(meas_1);
sigma_0 = 1/N_0 * sum((meas_0 - mu_0) * (meas_0 - mu_0)');
sigma_1 = 1/N_1 * sum((meas_1 - mu_1) * (meas_1 - mu_1)');

x = 72;
N = (N_0+N_1);
meas = [meas_0, meas_1];
mu = (mu_0 + mu_1) / 2;
shared_cov = 1/N * sum((meas - mu) * (meas - mu)');

likeli_0 = shared_cov ^ (-1/2) * exp(-1/2 * (x - mu_0) * 1/shared_cov * (x - mu_0));
likeli_1 = shared_cov ^ (-1/2) * exp(-1/2 * (x - mu_1) * 1/shared_cov * (x - mu_1));

tmp = likeli_0 * N_0 / (N_0 + N_1);
gda_pred = tmp / (tmp + likeli_1 * N_1 / (N_0 + N_1))
