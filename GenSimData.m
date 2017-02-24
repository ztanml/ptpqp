function [ Y ] = GenSimData( Alpha, THETA, Ncate, n, cont )
%% Generate n simulated data instances
% Alpha is the Dirichlet concentration parameters
% THETA is sum(Ncate)-by-k ground-truth matrix, Row 1:Ncate(1) correspond to
%     variable 1, row Ncate(1)+1:sum(Ncate(1:2)) correspond to variable 2, and so on.
% Ncate are the category counts of each variable
% Y is an n-by-p matrix whose i-th column corresponds to variable i and
%     takes the value in {0,1,...,Ncate(i)-1}
% cont is the probability of contamination, where a contaminated variable
%     is drawn from a discrete uniform distribution.

if nargin < 5
    cont = 0;
end

p = length(Ncate);

Y = zeros(p,n);
S = [0;cumsum(Ncate)];

for i = 1:n
    X = drchrnd(Alpha',1)';
    for j = 1:p
        P = (1 - cont) * THETA(S(j)+1:S(j+1),:)*X + cont*ones(Ncate(j),1)/Ncate(j);
        Y(j,i) = discreternd(P);
    end
end

end
