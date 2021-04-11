function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%   NNCOSTFUNCTION Implements the neural network cost function for a two layer
%   neural network which performs classification.
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.


% Reshape nn_params back into the parameters Theta1 and Theta2.
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));


m = size(X, 1);
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));


%Unregularize cost function
for i=1:m
  a1=X(i,:);
  a1=[ones(1,1) a1];
  a2=sigmoid(Theta1*a1');
  temp=a2';
  temp=[ones(1,1) temp];
  a3=sigmoid(Theta2*temp');
  
  logic=(1:num_labels);
  logic=(logic == y(i))';
  
  cost=sum(-1*logic.*log(a3)-(1-logic).*log(1-a3));
  J=J+cost;
  
endfor
J=J/m;
%Regularize cost function%

un_Theta1 = Theta1(:,2:end);
un_Theta2 = Theta2(:,2:end);

J = J+(lambda/(2*m))*(sum(sum(un_Theta1.^2))+sum(sum(un_Theta2.^2)));

%Backpropagation using for loop
delta1=0;
delta2=0;
Y=eye(num_labels)(y,:);
for i=1:m
  a1=X(i,:);
  a1=[ones(1,1) a1];
  a2=sigmoid(Theta1*a1');
  temp=a2';
  temp=[ones(1,1) temp];
  a3=sigmoid(Theta2*temp');
  
  new_y=Y(i,:)';
 
  sigma3=a3-new_y;
  sigma2=(un_Theta2' * sigma3) .* sigmoidGradient(Theta1*a1');
  
  delta2=delta2 + (sigma3 * temp);
  delta1=delta1 + (sigma2 * a1);
  
  
  
endfor
Theta1_grad=(1/m)*delta1;
Theta2_grad=(1/m)*delta2;


Theta1_grad = Theta1_grad + ((lambda/m)*[zeros(size(un_Theta1, 1), 1) un_Theta1]); 
Theta2_grad = Theta2_grad + ((lambda/m)*[zeros(size(un_Theta2, 1), 1) un_Theta2]); 


% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
