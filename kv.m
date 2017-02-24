function [ V ] = kv( a, d )
%% Computes a d-dimensional standard basis vector e_{a+1}


V = zeros(d,1);
V(a+1) = 1;

end
