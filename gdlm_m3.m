function [ T ] = gdlm_m3( Y, Idx, Alpha, Ncate )
%% Compute the empirical third-order estimator
% Y is the observation matrix
% Idx specifies the variables
% Alpha is the Dirichlet concentration parameter
% Ncate contains the number of categories of each variable

n = size(Y,2);

a0 = sum(Alpha);
beta = a0/(a0+2);
gamma = 2*a0*a0/(a0+1)/(a0+2);

T = zeros(Ncate(Idx(1)),Ncate(Idx(2)),Ncate(Idx(3)));

j = Idx(1);
s = Idx(2);
t = Idx(3);

uj = mkv(Y(j,:),Ncate(j));
us = mkv(Y(s,:),Ncate(s));
ut = mkv(Y(t,:),Ncate(t));
mts = gamma*tensor3_create(uj,us,ut);

% use tensor3_create to speedup arithmetics

for i = 1:n
    T = T + tensor3_create(kv(Y(j,i),Ncate(j)),kv(Y(s,i),Ncate(s)),kv(Y(t,i),Ncate(t)));
    T = T - beta*(tensor3_create(kv(Y(j,i),Ncate(j)),kv(Y(s,i),Ncate(s)),ut) + ...
        tensor3_create(uj,kv(Y(s,i),Ncate(s)),kv(Y(t,i),Ncate(t))) + ...
        tensor3_create(kv(Y(j,i),Ncate(j)),us,kv(Y(t,i),Ncate(t))));
end

T = T/n + mts;

end
