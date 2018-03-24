function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%

% 1. -- Expand the 'y' output values into a matrix of single values ---
eye_matrix = eye(num_labels);
y_matrix = eye_matrix(y,:);


% add column of 1's to the X matrix
X = [ones(size(X, 1), 1) X];

% The parameters for each unit in the neural network is represented in Theta1 and Theta2 as one
% row (the first row of Theta1 corresponds to the first hidden
% unit in the second layer. You can use a for-loop over the examples to
% compute the cost.

% 2a -- a1 equals X with bias unit
a1 = X;

% 2b. --  a2 equals the sigmoid of a1*theta
z2 = a1*Theta1';
a2 = sigmoid(a1*Theta1');
% add bias unit
a2_wBias = [ones(size(a2, 1), 1) a2];

% 2c. --  a3 equals the sigmoid of a2*theta
a3 = sigmoid(a2_wBias*Theta2');
h = a3;

% Cost function, non-regularized
Jpre = (-y_matrix.*log(h)) - ((1-y_matrix).*log(1-h));
J = sum(sum(Jpre))/m;
 
% Cost function, regularized

Theta1_noBias = Theta1(:,2:size(Theta1, 2));
Theta2_noBias = Theta2(:,2:size(Theta2, 2));

reg = sum(sum(Theta1_noBias.^2)) + sum(sum(Theta2_noBias.^2));
J = J + (lambda/(2*m)) * reg;


% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%

% this is assuming not doing with initialized weights
d3 = h-y_matrix;
d2 = d3*(Theta2(:,2:end)).*(sigmoidGradient(z2));

Delta1 = d2' * a1;
Delta2 = d3' * a2_wBias;

Theta1_grad = Delta1/m;
Theta2_grad = Delta2/m;

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

Theta1(:,1)=0;
Theta2(:,1)=0;

Theta1_grad = Theta1_grad + (lambda/m)*(Theta1);
Theta2_grad = Theta2_grad + (lambda/m)*(Theta2);


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
