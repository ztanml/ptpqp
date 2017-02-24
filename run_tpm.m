function [THETA] = run_tpm(M3, M2, Alpha, k, Ncate)
%% Run tensor power method.


p = size(Ncate,1);
N = sum(Ncate);
S = [0; cumsum(Ncate)];

THETA = zeros(N, k);

[V,~,~] = tpm_nonortho(tensor(M3), M2, Alpha, 50, 50, k);

V = abs(V);

for i = 1:p   
    THETA(S(i)+1:S(i+1),:) = normalize(V(S(i)+1:S(i+1),:),1);
end

end
