function [ B ] = gdlm_m2_full( Y, Alpha, Ncate )
%% Construct the full second-order estimator
% Y is the observation matrix
% Alpha is the Dirichlet concentration parameter
% Ncate contains the number of categories of each variable

p = size(Y,1);
N = sum(Ncate);

S = [0; cumsum(Ncate)];

% Create the block tensor
% The construction can be optimized. We use the simple method, 
% since we don't consider the construction time here
B = zeros(N,N);

for j = 1:p
    for s = 1:p
        B(S(j)+1:S(j+1),S(s)+1:S(s+1)) = gdlm_m2(Y, [j s]', Alpha, Ncate);
    end
end

end
