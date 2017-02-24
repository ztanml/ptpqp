function [ ThetaJ, ThetaS, ThetaT ] = frac_part( Y, PiJ, PiS, PiT, Alpha, k, Ncate, mode )
%% Learning a partition of GDLM paramters using TPQP
% Y is the observation matrix
% PiJ, PiS, PiT are vectors of variable indexes
% Alpha is the Dirichlet concentration parameter
% k is the number of hidden components
% Ncate contains the number of categories of each variable
% mode is either 'wnmf' or 'pqp'

pj = sum(Ncate(PiJ));
ps = sum(Ncate(PiS));
pt = sum(Ncate(PiT));

J = ones(length(PiJ)+1,1);
S = ones(length(PiS)+1,1);
T = ones(length(PiT)+1,1);

for i = 2:length(PiJ)+1
    J(i) = J(i-1) + Ncate(PiJ(i-1));
end
for i = 2:length(PiS)+1
    S(i) = S(i-1) + Ncate(PiS(i-1));
end
for i = 2:length(PiT)+1
    T(i) = T(i-1) + Ncate(PiT(i-1));
end

% Create the block tensor
B = zeros(pj,ps,pt);
for j = 1:length(PiJ)
    for s = 1:length(PiS)
        for t = 1:length(PiT)
            B(J(j):J(j+1)-1, S(s):S(s+1)-1, T(t):T(t+1)-1) = ...
                gdlm_m3(Y, [PiJ(j);PiS(s);PiT(t)], Alpha, Ncate);
        end
    end
end

[ThetaJ, ThetaS, ThetaT] = tpqp(B,k,'mode',mode);

for i = 1:length(PiJ)
    ThetaJ(J(i):J(i+1)-1,:) = normalize(ThetaJ(J(i):J(i+1)-1,:),1);
end
for i = 1:length(PiS)
    ThetaS(S(i):S(i+1)-1,:) = normalize(ThetaS(S(i):S(i+1)-1,:),1);
end
for i = 1:length(PiT)
    ThetaT(T(i):T(i+1)-1,:) = normalize(ThetaT(T(i):T(i+1)-1,:),1);
end

end

