function [J grad] = costFunction(X, y, theta, lambda)

m = length(y);
n = size(X, 2);

J = 0;
grad = zeros(n, 1);

h = sigmoid(X * theta);

J = (1/m) * ((-y)' * log(h) - (1-y)' * log(1-h) ...
   + (lambda/2)*sum((theta.^2)(2:end)));
   
grad = (1/m) * (X' * (h-y));
grad(2:end) = grad(2:end) + (lambda/m)*theta(2:end);

end