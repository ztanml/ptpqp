function [V, D, iters] = tpm_nonortho(T3, T2, Alpha, L, N, n)
%% Tensor power method for nonorthonormal factor matrices. The algorithm
%     first applies whitening (based on T2).
% T3 is \sum_i \alpha_i V_i x V_i x V_i tensor, where V is not necessarily orthogonal
% T2 is V x W x V^T
% n is the rank

% Whitening
[U,S,~] = svd(T2);
W = sqrt(pinv(S))*U';
T3 = ttm(T3,{W,W,W});

% Approximate orthogonal decomposition
[V,D,iters] = tpm(T3,L,N,n);

% Reverse whitening
a0 = sum(Alpha);
P  = sqrt(a0*(a0+1)./Alpha);
V  = U*sqrt(S)*V*diag(P);
D  = D(1:length(Alpha))./(P.^3);

end
