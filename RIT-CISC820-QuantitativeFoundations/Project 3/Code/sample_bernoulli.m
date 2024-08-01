function X = sample_bernoulli(N,theta)

X = double(rand(N,1)<theta);