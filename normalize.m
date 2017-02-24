function [ R ] = normalize( V, p )
%% Normalization using p-norm.


if nargin == 1
    p = 2;
end

R = bsxfun(@rdivide, V, sum(abs(V).^p).^(1/p));

end

