function [Hals] = run_ncp(M3, k, Ncate)
%% Nonnegative tensor factorizations Hals.
% Package NCP is available at: https://gist.github.com/panisson/7719245

p = size(Ncate,1);
N = sum(Ncate);
S = [0; cumsum(Ncate)];

Hals = zeros(N, k);

[M,~,~] = ncp(tensor(M3),k,'method','hals');
A = M.U{1};
for i = 1:p   
    Hals(S(i)+1:S(i+1),:) = normalize(A(S(i)+1:S(i+1),:),1);
end

end
