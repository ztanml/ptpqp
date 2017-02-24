%% Compute the error of MELD

K = [3 5 10 20];
N = [100 200 500 1000 5000];
C = [0 0.05 0.1];

fid = fopen('data/resultMELD.txt', 'wt');

for c = 1:length(C)
    for n = 1:length(N)
        for k = 1:length(K)
            Theta_true = load(['data/' num2str(K(k)) 'n' num2str(N(n)) ...
                'c' num2str(C(c)) 'Theta.txt']);
            Theta_MELD = load(['data/MELDd4p25k' num2str(K(k)) 'n' num2str(N(n)) ...
                'c' num2str(C(c)) '.txt']);
            fprintf(fid, 'n=%d k=%d c=%f: %f\n', ...
                N(n), K(k), C(c), CompErr(Theta_true,Theta_MELD));
        end
    end
end

fclose(fid);
