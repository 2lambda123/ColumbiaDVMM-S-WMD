function [Anew] = get_nca_matrix(X,Y,output_dim)
num_class = max(Y);
[Y,index] = sort(Y);
X = X(:,index);
N_point = size(X,2);

YMatrix = zeros(num_class,N_point);

for i = 1:num_class
    YMatrix(i,Y==i) = 1;
end

input_dim = size(X,1);

A = zeros(output_dim,input_dim);

for i = 1:output_dim
    A(i,i) = 1;
end

[Anew,fX,i] = minimize(A(:), 'nca_obj', 100, X', YMatrix');

Anew = reshape(Anew,output_dim,input_dim);
