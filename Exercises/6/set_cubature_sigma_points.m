function xs = set_cubature_sigma_points(m,P,n)
    xs = zeros(n,2*n);
    L = chol(P,'lower');
    indices = (1:n)*2;
    for j=1:n
        xs(:,indices(j)-1) = m + sqrt(n)*L(:,j);
        xs(:,indices(j)) = m - sqrt(n)*L(:,j);
    end
end

