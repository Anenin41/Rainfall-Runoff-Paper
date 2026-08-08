function W = initial_condition(x,N,n,u,h)
    % Input variables

    % x         = input domain
    % N         = total number of equations
    % n         = number of moments
    % dt        = time interval
    % h         = water height
    % u         = mean velocity

    % Output 
    % W         = initial conditions of all variables 

W  = zeros (N,length(x));
    if (n==0)
         um = integral(u,0,1).*ones(size(x));
         hum= h.*um;
       
        % Assigning Initial condition
        W(1,:) = h; W(2,:) = hum; 
    
    
    else
        Alpha  = initial_momemt(N,u);
        um = integral(u,0,1).*ones(size(x));
        hum= h.*um;
    
        % Assigning Initial condition
        W(1,:) = h; W(2,:) = hum; 
        W(3:N,:) = h.*Alpha;
    
    end
end