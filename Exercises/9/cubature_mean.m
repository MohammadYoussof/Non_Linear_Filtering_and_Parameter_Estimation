function mean = cubature_mean(xhat, nsp)
    [a,~] = size(xhat);
    mean = zeros(a,1);
    for j=1:nsp
        mean = mean + xhat(:,j);
    end
    mean = mean/nsp;
end

