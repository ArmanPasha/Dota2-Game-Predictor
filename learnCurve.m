function [train_error test_error] = learnCurve(X, y, X_test, y_test, lambda)
  %Iterate over different number of m to return the test and train error
  %to check the performance
  m = size(X, 1);
  train_error = zeros(m, 1);
  test_error = zeros(m, 1);
  
  for i=1:m
    %Shuffle X for randomness
    %X = X(randperm(m), :);
    [theta] = trainer(X([1:i], :), y([1:i]), lambda, 50);
    train_error(i) = costFunction(X([1:i], :), y([1:i]), theta, 0);
    test_error(i) = costFunction(X_test, y_test, theta, 0);
  end
  
endfunction
