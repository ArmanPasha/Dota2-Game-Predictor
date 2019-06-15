clear; clc; close;
data = load('dataset.txt');
X = data(:, [1:3]);
y = data(:, 4);

m = length(y);

%Data is recenlty sorted, so we shuffle
shuffle = randperm(m);
X = X(shuffle, :);
y = y(shuffle);

lambda = 0.03;

%Add bias column to X
X = [ones(m, 1) X];
%mu and sigma are column vectors; normalize for better performance
[X_norm mu sigma] = normalize(X);

%Split data to training set and test set 
train_num = floor(0.7*m);
X_train = X_norm([1:train_num], :);

X_test = X_norm([train_num + 1 : end], :);
y_test = y([train_num + 1 : end]);

y_train = y([1:train_num]);

%   a a a
% x a a a
% x a a a h        Neural Network Architecture (without bias units)
% x a a a
%   
%

arc_lengths = [3; 4; 4; 4; 1];
max_iter = 200;
%initializing theta with random variables (it must be random)
%theta is unrolled
init_theta = rand((arc_lengths(1)+1)*arc_lengths(2) ...
           + (arc_lengths(2)+1)*arc_lengths(3) ...
           + (arc_lengths(3)+1)*arc_lengths(4) ...
           + (arc_lengths(4)+1)*arc_lengths(5) ,1);
           
cf = @(t) NNcostFunction(X_train, y_train, t, arc_lengths, lambda);

options = optimset('MaxIter', max_iter);
theta = fmincg(cf, init_theta, options);

h_train = NNpredict(X_train, theta, arc_lengths);
printf("Accuracy over training set: %f \n", mean(y_train==h_train) * 100);

h_test = NNpredict(X_test, theta, arc_lengths);
printf("Accuracy over test set: %f \n", mean(y_test==h_test) * 100);











