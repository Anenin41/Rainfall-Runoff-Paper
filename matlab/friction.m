function S_w = friction(nx,N,nu,slip_length,w)
    % Input variables
    % nx        = total number of grids
    % N         = total number of equations
    % nu        = scaled kinematic viscosity
    % slip_length
    % w         = initial variables h, u and alphas

    % Output 
    % S_w       = source term on the rhs of the PDE


    % Calculate the friction term
    n= N-2; %number of moments
    % Initialization of friction matrix
    S_w   = zeros (N,nx);
    % Initialization of moment coefficients
    a     = zeros (n+1,nx);  

    c     = zeros(n+1,n);
    S_P   = zeros (n+1,nx);
    for i= 1:n+1
        for j=1:n
            if mod(i+j,2)==0
                c(i,j)= 0;
            else
                c(i,j)= 0.5.*(min(i-1,j)).*(min(i-1,j)+1);
            end
        end
    end
    %  Calculate moment coefficient \alpha h*alpha/h
    for k = 1: n
        a(k+1,:) = w(k+2,:)./w(1,:);
    end
    %  Calculate velocity h*um/h
    u(1,1:nx) = w(2,:)./w(1,:);
    
    % second part of the equation \sum_1^N c_{i,j}\alpha_j
    for i = 1:nx
        S_P(2:end,i) =   sum(c.*a(:,i));
    end
    % Calcuate the friction term
    for i= 1:n+1
        for j = 1:nx
            P(i,j)=   - ((nu/slip_length)*(2*i-1).*(u(j) + sum(a(2:end,j)) ))- (nu./w(1,j)).*4.*(2*i-1).* S_P(i,j) ;
        end
    end
    S_w(2:end,:) = P;
end
