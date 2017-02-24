%K = [3 5 10 20];
%N = [100 200 500 1000 5000];
%C = [0 0.05 0.1];
K = [3 5 10 20];
N = [100 200 500 1000 5000];
C = [0.1];
p = 10;

for c = 1:length(C)
    for k = 1:length(K)
        for n = 1:length(N)
            % Generate data
            Theta_true = GenTheta(ones(p,1)*4,K(k));
            Y = GenSimData(ones(K(k),1)*0.1, Theta_true, ones(p,1)*4, N(n), C(c));
            M = gdlm_m3_full(Y,ones(K(k),1)*0.1,ones(p,1)*4);
            fprintf('n=%d k=%d c=%f: %f\n', N(n), K(k), C(c), ...
                sum(sum(sum(M<0)))/numel(M));
        end
    end
end
