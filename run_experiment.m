function [pqp, tpm, nojd0, nojd1, hals] = run_experiment(k,n,cont,Alpha,Ncate,tries)
%% Run experiment with synthetically generated data
% Example parameter settings:
% k = 3;  % number of hidden components
% n = 1000; % number of observations
% cont = 0; % percent of contamination
% Alpha = [0.1 0.2 0.3]'; % Dirichlet concentration over hidden components
% Ncate = ones(25,1) * 4; % 25 variables, each variable takes categorical values in {0,1,2,3}
% tries = 5; % use the average of a number of tries

addpath('nonnegfac-matlab/');

err_pqp = 0;
err_tpm = 0;
err_nojd0 = 0;
err_nojd1 = 0;
err_hals = 0;

for i = 1:tries
    % Generate data
    Theta_true = GenTheta(Ncate,k);
    Y = GenSimData(Alpha, Theta_true, Ncate, n, cont);

    % Save the data for running MELD (python scripts)
    save(['data/k' num2str(k) 'n' num2str(n) 'c' num2str(cont) '.txt'], 'Y', '-ascii');
    save(['data/k' num2str(k) 'n' num2str(n) 'c' num2str(cont) 'Theta.txt'], 'Theta_true', '-ascii');    
    
    % Run tensor PQP method with 
    Theta_pqp = ptpqp(Y, Alpha, k, 1, 2, 3, 1, Ncate, ...
        'matching', 'opm', 'mode', 'pqp');
    err_pqp = err_pqp + CompErr(Theta_true, Theta_pqp);
    
    M3 = gdlm_m3_full(Y, Alpha, Ncate);
    M2 = gdlm_m2_full(Y, Alpha, Ncate);

    % Run tpm
    Theta_tpm = run_tpm(M3, M2, Alpha, k, Ncate);
    err_tpm = err_tpm + CompErr(Theta_true, Theta_tpm);
    
    % Run nojd0 and nojd1
    [Theta_nojd0, Theta_nojd1] = run_nojd(M3, k, Ncate);
    err_nojd0 = err_nojd0 + CompErr(Theta_true, Theta_nojd0);
    err_nojd1 = err_nojd1 + CompErr(Theta_true, Theta_nojd1);
    
    % Run hals
    Theta_hals = run_ncp(M3, k, Ncate);
    err_hals = err_hals + CompErr(Theta_true, Theta_hals);
end

pqp = err_pqp/tries;
tpm = err_tpm/tries;
nojd0 = err_nojd0/tries;
nojd1 = err_nojd1/tries;
hals = err_hals/tries;

end
