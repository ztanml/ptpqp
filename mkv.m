function [ Ac ] = mkv( A, d )
%% Compute the mean of d-dimensional standard basis vectors: \sum_i^n e_{A_i + 1} / n


Ac = zeros(d,1);

for i = 1:length(A)
    Ac(A(i)+1) = Ac(A(i)+1) + 1;
end

Ac = Ac/length(A);

end
