function [ M ] = gdlm_m2( Y, Idx, Alpha, Ncate )
%% Compute the empirical second-order estimator
% Y is the observation matrix
% Idx specifies the variables
% Alpha is the Dirichlet concentration parameter
% Ncate contains the number of categories of each variable

n = size(Y,2);

a0 = sum(Alpha);
beta = a0/(a0+1);

j = Idx(1);
t = Idx(2);

M = zeros(Ncate(j),Ncate(t));

uj = mkv(Y(j,:),Ncate(j));
ut = mkv(Y(t,:),Ncate(t));
mts = beta*uj*ut';

for i = 1:n
    M = M + kv(Y(j,i),Ncate(j))*kv(Y(t,i),Ncate(t))';
end

M = M/n - mts;

end
