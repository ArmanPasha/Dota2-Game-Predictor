function [theta] = trainer(X, y, lambda, max_iter = 200)
  
  initial_theta = zeros(size(X, 2), 1);
  costFunction1 = @(t) costFunction(X, y, t, lambda);
  options = optimset('MaxIter', max_iter, 'GradObj', 'on');
  theta = fmincg(costFunction1, initial_theta, options);
  
end
