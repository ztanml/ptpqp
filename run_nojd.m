function [THETA0, THETA1] = run_nojd(M3, k, Ncate)
%% Run Kuleshov's algorithms.
% THETA0 and THETA1 are the results of nojd0 and nojd1 algorithms.


p = size(Ncate,1);
S = [0; cumsum(Ncate)];

[THETA1, ~, misc] = no_tenfact(tensor(M3), 2*size(M3,1), k);
THETA1 = abs(THETA1);
THETA0 = abs(misc.V0);

for i = 1:p   
    THETA0(S(i)+1:S(i+1),:) = normalize(THETA0(S(i)+1:S(i+1),:),1);
    THETA1(S(i)+1:S(i+1),:) = normalize(THETA1(S(i)+1:S(i+1),:),1);
end

end
