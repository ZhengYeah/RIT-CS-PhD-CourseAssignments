% plot beta prior
alpha = 2;
beta = 2;
theta = 0:0.01:1;
y1 = betapdf(theta, alpha, beta);
% plot bernoulli likelihood
y2 = theta.^ 4 .* (1 - theta);
% plot posterior
y3 = betapdf(theta, 6, 3);


hold on;
plot(theta,y1, 'DisplayName','Prior','LineWidth', 2)
plot(theta,y2, 'DisplayName','Likelihood','LineWidth', 2)
plot(theta,y3, 'DisplayName','Posterior','LineWidth', 2)

fontname(gca,"Times New Roman")
xlabel('Parameter \theta') 
ylabel('Probability') 
hold off;

legend
