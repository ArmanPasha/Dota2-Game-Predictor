function [h] = predict(X, theta)
  h = round(sigmoid(X * theta));
endfunction
