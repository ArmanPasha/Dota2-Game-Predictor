function [J grad] = NNcostFunction(X, y, theta, arc_lengths, lambda)
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
                    
  J = 0;
  Theta1_grad = zeros(size(Theta1));
  Theta2_grad = zeros(size(Theta2));
  Theta3_grad = zeros(size(Theta3));
  Theta4_grad = zeros(size(Theta4));
  
  %forward propagation
  a2 = sigmoid(Theta1 * X'); %4*m
  a2 = [ones(1, size(a2, 2)); a2]; %5*m
  a3 = sigmoid(Theta2 * a2); %4*m
  a3 = [ones(1, size(a3, 2)); a3]; %5*m
  a4 = sigmoid(Theta3 * a3); %4*m
  a4 = [ones(1, size(a4, 2)); a4]; %5*m
  h = sigmoid(Theta4 * a4); %1*m
  h = h'; %m*1
  
  %Theta sum for regularization without including theta's correspending to bias terms
  Theta_sum = sum(((Theta1.^2)(:, 2:end))(:)) ...
            + sum(((Theta2.^2)(:, 2:end))(:)) ...
            + sum(((Theta3.^2)(:, 2:end))(:)) ...
            + sum(((Theta4.^2)(:, 2:end))(:));
  
  J = ((-1/m) * (y' * log(h) + (1-y)' * log(1-h))) + ...
         (lambda/(2*m)) * Theta_sum ;
         
  %backpropagation
  Delta1 = zeros(size(Theta1));
  Delta2 = zeros(size(Theta2));
  Delta3 = zeros(size(Theta3));
  Delta4 = zeros(size(Theta4));
  
  sigma5 = (h - y)'; %1*m
  sigma4 = (Theta4' * sigma5) .* a4 .* (1-a4); %5*m
  %removing bias term
  sigma4 = sigma4(2:end, :); %4*m
  sigma3 = (Theta3' * sigma4) .* a3 .* (1-a3); %5*m
  sigma3 = sigma3(2:end, :); %4*m
  sigma2 = (Theta2' * sigma3) .* a2 .* (1-a2); %5*m
  sigma2 = sigma2(2:end, :); %4*m
  
  Delta1 = sigma2 * X; %4*4
  Delta2 = sigma3 * a2'; %4*5
  Delta3 = sigma4 * a3'; %4*5
  Delta4 = sigma5 * a4'; %1*5
  
  Theta1_grad = Delta1/m;
  Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + lambda*Theta1(:, 2:end);
  
  Theta2_grad = Delta2/m;
  Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + lambda*Theta2(:, 2:end);
  
  Theta3_grad = Delta3/m;
  Theta3_grad(:, 2:end) = Theta3_grad(:, 2:end) + lambda*Theta3(:, 2:end);
  
  Theta4_grad = Delta4/m;
  Theta4_grad(:, 2:end) = Theta4_grad(:, 2:end) + lambda*Theta4(:, 2:end);
  
  grad = [Theta1_grad(:); Theta2_grad(:); Theta3_grad(:); Theta4_grad(:)];
  
  
endfunction
