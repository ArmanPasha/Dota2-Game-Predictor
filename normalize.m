function [X_norm mu sigma] = normalize(X)
  %Assumption: X already has the bias column of ones
  X_norm = X;
  n = size(X,2);
  mu = zeros(n,1);
  sigma = zeros(n,1);
  for i=2:n
    mu(i) = mean(X(:, i));
    sigma(i) = std(X(:, i));
    X_norm(:, i) = (X_norm(:, i) - mu(i))/sigma(i);
  end
  
endfunction
