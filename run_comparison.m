%% Comparing tensor algorithms for learning GDLMs.
% Note that the other tensor methods take a long time to run. Consider
% parallelizing this script on multiple machines with disjoint
% experiment configurations.

%% Comparing the tensor algorithms under all combinations of the following configurations
% Example configurations:
% K = [3 5 10 20];
% N = [100 200 500 1000 5000];
% C = [0 0.05 0.1];

tries = 1; % number of tries, you may use 1 for a quick check

K = [3];   % Number of hidden components to test
N = [100]; % Number of samples for computing the empirical tensor
C = [0];   % Contamination levels

for c = 1:length(C)
    for k = 1:length(K)
        for n = 1:length(N)
            [pqp, tpm, nojd0, nojd1, hals] = run_experiment(K(k),N(n),C(c),ones(K(k),1)*0.1,ones(25,1)*4,tries);
            fid = fopen('data/result.txt', 'at');
            fprintf(fid, 'n=%d k=%d c=%f: %f %f %f %f %f\n', ...
                N(n), K(k), C(c), pqp, tpm, nojd0, nojd1, hals);
            fclose(fid);
        end
    end
end
