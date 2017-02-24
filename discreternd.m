function x = discreternd(p,n)
%% Generate @n discrete random values using given probability vector @p

if nargin == 1
    n = 1;
end

k = length(p);
x = sum(repmat(rand(n,1),1,k)> repmat(cumsum(p)'/sum(p),n,1),2);

end
