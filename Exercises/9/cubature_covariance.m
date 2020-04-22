function P = cubature_covariance(xhat, mm, yhat, mu, nsp)
    [a,~] = size(xhat);
    [b,~] = size(yhat);
    P = zeros(a,b);
    for j=1:nsp
        P = P + (xhat(:,j)-mm)*(yhat(:,j)-mu)';
    end
    P = P/nsp;
end

