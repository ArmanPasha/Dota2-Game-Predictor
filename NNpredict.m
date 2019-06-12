function [h] = NNpredict(X, theta, arc_lengths)
  %arc_lengthss is a vector containing the size of each layer in NN
  %arc_lengths = [3; 4; 4; 4; 1]
  %theta is an unrolled vector containing all of the Theta matrices
  %ts = theta size
  ts1 = (arc_lengths(1) + 1) * arc_lengths(2);
  Theta1 = reshape(theta(1:ts1), arc_lengths(2), arc_lengths(1) + 1); %4*4
  
  ts2 = (arc_lengths(2) + 1) * arc_lengths(3);
  Theta2 = reshape(theta((ts1 + 1):(ts1 + ts2)), arc_lengths(3), arc_lengths(2) + 1);%4*5
  
  ts3 = (arc_lengths(3) + 1) * arc_lengths(4);
  Theta3 = reshape(theta((ts1 + ts2 + 1):(ts1 + ts2 + ts3)), ...
                   arc_lengths(4), arc_lengths(3) + 1); %4*5
                   
  ts4 = (arc_lengths(4) + 1) * arc_lengths(5);
  Theta4 = reshape(theta((ts1 + ts2 + ts3 + 1):(ts1 + ts2 + ts3 + ts4)), ...
                    arc_lengths(5), arc_lengths(4) + 1);    %1*5
                    
  m = size(X, 2); %number of training examples
  
  %forward propagation
  a2 = sigmoid(Theta1 * X'); %4*m
  a2 = [ones(1, size(a2, 2)); a2]; %5*m
  a3 = sigmoid(Theta2 * a2); %4*m
  a3 = [ones(1, size(a3, 2)); a3]; %5*m
  a4 = sigmoid(Theta3 * a3); %4*m
  a4 = [ones(1, size(a4, 2)); a4]; %5*m
  h = sigmoid(Theta4 * a4); %1*m
  h = round(h');
endfunction
