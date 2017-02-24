function [ THETA ] = ptpqp( Y, Alpha, k, PiJ, PiS, PiT, npart, Ncate, varargin)
%% Learning a GDLM using partitioned tensor parallel quadratic programming (PTPQP)
% Y is the observation matrix
% Alpha is the Dirichlet concentration parameters
% k is the number of hidden components
% PiJ, PiS, PiT are index vectors for the common variables
% npart is the number of partitions
% Ncate contains the number of categories for each variable
% Other options:
%   matching: either 'opm' or 'sam', default is 'opm'
%   mode: either 'wnmf' or 'pqp'

opt = inputParser;
opt.addParameter( 'matching', 'opm',  @(x) ischar(x) & (strcmp(x,'opm')|strcmp(x,'sam')) );
opt.addParameter( 'mode',     'wnmf', @(x) ischar(x) );
opt.parse(varargin{:});
opt = opt.Results;

use_opm = strcmp(opt.matching, 'opm');

Refvars = unique([PiJ;PiS;PiT]);

p = size(Y,1) - length(Refvars);
m = floor(p/npart/3);

if m < 1
    if p >= 3
        error('Too many partitions.');
    else
        warning('Too few variables, will use only one partition.')
        npart = 1;
    end
end

V2p = setdiff((1:size(Y,1))', Refvars);
V2p = V2p(randperm(p));

THETA = zeros(sum(Ncate), k);

S = [0; cumsum(Ncate)] + 1;

for i = 1:npart
    from = 3*(i-1)*m;
    Aidx = V2p(from+1:from+m);
    Bidx = V2p(from+m+1:from+2*m);
    
    if i == npart
        Cidx = V2p(from+2*m+1:end);
    else
        Cidx = V2p(from+2*m+1:from+3*m);
    end
    
    [A, B, C] = frac_part(Y, [Aidx;PiJ], [Bidx;PiS], [Cidx;PiT], Alpha, k, Ncate, opt.mode);
    
    R1 = A(sum(Ncate(Aidx))+1:end,:);
    R2 = B(sum(Ncate(Bidx))+1:end,:);
    R3 = C(sum(Ncate(Cidx))+1:end,:);
    
    if i == 1
        R = normalize([R1; R2; R3]); % reference matrix
        % Write paramters of reference variables
        where = 1;
        for j = 1:length(PiJ)
            THETA(S(PiJ(j)):S(PiJ(j))+Ncate(PiJ(j))-1,:) = R1(where:where+Ncate(PiJ(j))-1,:);
            where = where + Ncate(PiJ(j));
        end
        where = 1;
        for j = 1:length(PiS)
            THETA(S(PiS(j)):S(PiS(j))+Ncate(PiS(j))-1,:) = R2(where:where+Ncate(PiS(j))-1,:);
            where = where + Ncate(PiS(j));
        end
        where = 1;
        for j = 1:length(PiT)
            THETA(S(PiT(j)):S(PiT(j))+Ncate(PiT(j))-1,:) = R3(where:where+Ncate(PiT(j))-1,:);
            where = where + Ncate(PiT(j));
        end
    else
        % Compute a permutation
        Q = normalize([R1; R2; R3]);
        if ~use_opm
            [~,Psi] = max(Q'*R);
            % Check the sufficient condition
            if length(unique(Psi)) ~= k
                warning('SAM yielded permutation with duplicates.');
            end
            Q = Q(:,Psi);
            mca = min(diag(Q'*R)); % cosine of maximum perturbation angle
            sca = sqrt(sqrt((1+max(max(Q'*Q - eye(k))))/8) + 1/2); % cosine bound from sufficient cond
            if mca > sca
                disp('Guaranteed SAM for part ' + string(i));
            else
                disp('Suff cond does not hold: ' + string(mca) + ' <= ' + string(sca));
            end
        else
            [U,~,V] = svd(Q'*R);
            [~,Psi] = max(U*V');
            % We may also check the sufficient condition
            if length(unique(Psi)) ~= k
                warning('OPM yielded permutation with duplicates.');
            end
            Q = Q(:,Psi);
            E = Q'*(Q - R);
            SE = svd(E);
            SQ = svd(Q'*Q);
            if max(SE) >= min(SQ)
                disp('Angle spectral norm is large: ' + ...
                    string(max(SE)) + ' >= ' + string(min(SQ)));
            else
                rho = sum(SE(1:2));
                nu  = sum(SQ(end-1:end));
                if -max(SE)/rho*log(1-rho/nu) >= (2-sqrt(2))/4
                    disp('Suff cond does not hold: ' + string(-max(SE)/rho*log(1-rho/nu)) + ...
                        ' >= ' + string((2-sqrt(2))/4));
                else
                    disp('Guaranteed OPM for part ' + string(i));
                end
            end
        end
        A = A(:,Psi);
        B = B(:,Psi);
        C = C(:,Psi);
    end
    
    % Write paramters of partition variables to THETA
    where = 1;
    for j = 1:length(Aidx)
        THETA(S(Aidx(j)):S(Aidx(j))+Ncate(Aidx(j))-1,:) = A(where:where+Ncate(Aidx(j))-1,:);
        where = where + Ncate(Aidx(j));
    end
    where = 1;
    for j = 1:length(Bidx)
        THETA(S(Bidx(j)):S(Bidx(j))+Ncate(Bidx(j))-1,:) = B(where:where+Ncate(Bidx(j))-1,:);
        where = where + Ncate(Bidx(j));
    end
    where = 1;
    for j = 1:length(Cidx)
        THETA(S(Cidx(j)):S(Cidx(j))+Ncate(Cidx(j))-1,:) = C(where:where+Ncate(Cidx(j))-1,:);
        where = where + Ncate(Cidx(j));
    end
end

end
