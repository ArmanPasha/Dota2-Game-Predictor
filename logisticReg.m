clear; clc; close all;
data = load('dataset.txt');
X = data(:, [1:3]);
y = data(:, 4);

m = length(y);

lambda = 0;

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

%Computing the theta
[theta] = trainer(X_train, y_train, lambda)

%Draw the learn curve
[train_error test_error] = learnCurve(X_train, y_train, X_test, y_test, lambda);
figure;
plot([1:train_num], train_error, [1:train_num], test_error);

%Plot data
x1 = X_norm(:, 2); %Score
x2 = (X_norm(:, 3) + X_norm(:, 4))/2; %Avg of gold and XP
figure;
plot(x1(y==1), x2(y==1), 'rx', 'MarkerSize', 20 ...
      , x1(y==0), x2(y==0), 'bo', 'MarkerSize', 20);
xlabel('Score diff');
ylabel('Avg gold & XP diff');
hold on;


u = linspace(min(X_norm(:, 2)), max(X_norm(:, 2)), 100);
v = linspace(-2, 2, 100);
Z = zeros(length(v), length(u));
for i=1:length(u)
  for j=1:length(v)
    Z(j, i) = [1 u(i) v(j) v(j)] * theta;
  end
end
contour(u, v, Z, [0, 0], 'LineWidth', 4)

%Accuracy on training set and test set
h_train = predict(X_train, theta);
fprintf("Accuracy over training set is %f\n", mean(y_train==h_train)*100);

h_test = predict(X_test, theta);
fprintf("Accuracy over test set is %f\n", mean(y_test==h_test)*100);