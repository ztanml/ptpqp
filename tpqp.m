function [ A, B, C ] = tpqp( T, r, varargin )
%% Tensor PQP algorithm for factorizing three-way tensors
% T: three-way tensor
% r: rank
% Other options:
%   mode: can be either 'pqp' or 'wnmf', default is 'pqp'
%   tol: stopping tolerance, smaller tol and larger max_iter for more accurate factorization
%   scaling: use scaling to speedup factorization
%   max_iter: maxium number of iterations

% Parse options
opt = inputParser;
opt.addParameter( 'mode',     'pqp',  @(x) ischar(x) & (strcmp(x,'pqp')|strcmp(x,'wnmf')) );
opt.addParameter( 'tol',      1e-12,  @(x) isscalar(x) & x > 0 );
opt.addParameter( 'scaling',  true,   @(x) islogical(x) );
opt.addParameter( 'max_iter', 50000,  @(x) isscalar(x) & x > 0 );
opt.parse(varargin{:});
opt = opt.Results;

% Scaling the tensor to speedup factorization
if opt.scaling
    sc = max(max(max(abs(T))));
    T  = T/sc;
end

A = rand(size(T,1), r);
B = rand(size(T,2), r);
C = rand(size(T,3), r);

M1 = double(tenmat(T,1));
M2 = double(tenmat(T,2));
M3 = double(tenmat(T,3));

old_err = 0;

niter = opt.max_iter;
mode  = strcmp(opt.mode,'pqp');

if ~mode
    L1 = M1>0;
    L2 = M2>0;
    L3 = M3>0;
    M1 = max(M1,0);
    M2 = max(M2,0);
    M3 = max(M3,0);
end

while niter ~= 0
    if mode  % PQP
        A = A .* pqp_mult(A,(C'*C).*(B'*B),M1*khatrirao(C,B),opt.tol);
        B = B .* pqp_mult(B,(C'*C).*(A'*A),M2*khatrirao(C,A),opt.tol);
        C = C .* pqp_mult(C,(B'*B).*(A'*A),M3*khatrirao(B,A),opt.tol);
    else % WNMF
        A = A .* wnmf_mult(A,M1,khatrirao(C,B),L1,opt.tol);
        B = B .* wnmf_mult(B,M2,khatrirao(C,A),L2,opt.tol);
        C = C .* wnmf_mult(C,M3,khatrirao(B,A),L3,opt.tol);
    end
    cur_err = norm(M1-A*khatrirao(C,B)','fro');
    if old_err > 0 && old_err - cur_err < opt.tol
        break;
    end
    old_err = cur_err;
    niter = niter - 1;
end

if opt.scaling
    C = sc*C;
end

end

function Y = pqp_mult(X,Q,Z,tol)
    eigmin = min(eig(Q));
    if eigmin > 0
        PHI = max(sqrt(diag(Z*inv(Q)*Z')*diag(Q)'/eigmin) - abs(Z), 0)/2 + tol;
    else
        PHI = tol;
    end
    Y = (max(Z,0) + PHI) ./ (X*Q + max(-Z,0) + PHI);
end

function Y = wnmf_mult(X,U,W,L,tol)
    Y = (U*W + tol) ./ (((X*W').*L)*W + tol);
end
