function ri = resample(w)
% Systematic resampling
%
% SYNOPSIS
%   ri = resample(w)
%
% DESCRIPTION
%   Systematic resampling algorithm, i.e. returns indices such that the
%   probability of each index is approximately equal to the weight
%   specified for that index.
%
% PARAMETERS
%   w   Vector of probabilities
%
% RETURNS
%   ri  Indices
%
% AUTHOR
%   2016-03-14 -- Roland Hostettler <roland.hostettler@aalto.fi>

    M = length(w);
    ri = zeros(1, M);
    i = 0;
    u = 1/M*rand();
    for j = 1:M
        Ns = floor(M*(w(j)-u)) + 1;
        if Ns > 0
            ri(i+1:i+Ns) = j;
            i = i + Ns;
        end
        u = u + Ns/M - w(j);
    end;
    ri = ri(randperm(M));
end
