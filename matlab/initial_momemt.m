function Alpha = initial_momemt(N,u)

syms z
n = N-2;
    for j= 1:n
        phi(j,1)= (1/factorial(j)).*diff ((z-z.^2).^j,j);
        Alpha(j,1) = (2*j+1)* (int(u.* phi(j,1),[0 1]));
    end
end
