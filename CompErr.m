function [ err ] = CompErr( Theta_true, Theta )
%% Compute the minimum RMSE that takes into account column permutations

R = normalize(Theta_true);
Q = normalize(Theta);
[U,~,V] = svd(Q'*R);
[~,Psi] = max(U*V');
Theta = Theta(:,Psi);

err = norm(Theta_true - Theta,'fro') / sqrt(numel(full(Theta)));

end
