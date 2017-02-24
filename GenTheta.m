function [ Theta ] = GenTheta( Ncate, k )
%% Generate ground-truth Theta from a Dirichlet discribution.
% Ncate gives the number of categories for each variable

alpha = 0.5;

S = [0; cumsum(Ncate)];

Theta = zeros(sum(Ncate), k);

for i = 1:length(Ncate)
    Theta(S(i)+1:S(i+1),:) = drchrnd(ones(1,Ncate(i))*alpha, k)';
end

end

