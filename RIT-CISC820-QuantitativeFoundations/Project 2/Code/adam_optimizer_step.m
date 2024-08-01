function res = adam_optimizer_step(f_gradient, x_0, alpha, eps, beta_1, beta_2)

    % remember clear adam_optimizer_step after usage
    persistent k;
    persistent m;
    persistent v;
    persistent beta1_;
    persistent beta2_;
    
    grad = f_gradient(x_0);

    if isempty(k)
        k = 1;
        m = grad;
        v = grad.^2;
        beta1_ = beta_1;
        beta2_ = beta_2;
    else
        k = k + 1;
        m = beta_1 * m + (1 - beta_1) * grad;
        v = beta_2 * v + (1 - beta_2) * grad.^2;
        beta1_ = beta1_ * beta_1;
        beta2_ = beta2_ * beta_2;
    end
    
    m_hat = m / (1 - beta1_);
    v_hat = v / (1 - beta2_);
    
    res = alpha * m_hat ./ (sqrt(v_hat) + eps);
end