% plot Gaussian prior
x = -5:0.1:10;
y1 = normpdf(x, 0, 6);
% plot likelihood
data = [
3.2877564  
3.3049376 
3.0065537
2.4179332  
2.4255153 
3.3632328  
3.3750101 
3.6351869 
2.5573695  
2.4708438  
2.7914828  
2.8363451  
3.5001336
2.7917707  
2.3019892 
2.8733925  
2.4083626  
2.9122397  
3.1874000  
3.3319748];

hat_x = mean(data);
hat_x_expand = repmat(hat_x, 20, 1);
hat_sigma = immse(data, hat_x_expand);
sigma_2 = 10;

y2 = exp(-1 / (2 * sigma_2) * (20 * hat_sigma + 20 * (hat_x - x).^2));
% plot posterior
y3 = normpdf(x, 2.7130, 6/13);

% posteriot pred
y_4 = normpdf(2.4, 2.7130, 6/13 + 10);


hold on;
plot(x,y1, 'DisplayName','Prior','LineWidth', 2)
plot(x,y2, 'DisplayName','Likelihood','LineWidth', 2)
plot(x,y3, 'DisplayName','Posterior','LineWidth', 2)

fontname(gca,"Times New Roman")
xlabel('Parameter x') 
ylabel('Probability') 
hold off;

legend
