clear; clc; close;
data = load('dataset.txt');
X = data(:, [1:3]);
y = data(:, 4);

m = length(y);

lambda = 0;

%Add bias column to X
X = [ones(m, 1) X];
%mu and sigma are column vectors
[X_norm mu sigma] = normalize(X);
%Split data to training set and test set 
train_num = floor(0.7*m);
X = X_norm([1:train_num], :);

X_test = X_norm([train_num + 1 : end], :);
y_test = y([train_num + 1 : end]);

y = y([1:train_num]);

%Computing the theta
[theta] = trainer(X, y, lambda)

%Draw the learn curve
[train_error test_error] = learnCurve(X, y, X_test, y_test, lambda);
plot([1:train_num], train_error, [1:train_num], test_error);

%Accuracy on training set and test set
h_train = predict(X, theta);
fprintf("Accuracy over training set is %f\n", mean(y==h_train));

h_test = predict(X_test, theta);
fprintf("Accuracy over test set is %f\n", mean(y_test==h_test));