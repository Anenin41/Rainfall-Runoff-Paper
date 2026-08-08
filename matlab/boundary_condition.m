function [A,D_edge_minus_nx,  D_edge_plus_1] = boundary_condition(test_case,w,x,dx,dt,N,nx,g,matrix)
    % Input variables
    % test case = user input either Smooth_periodic or Dam-break
    % w         = initial variables h, u and alphas
    % x         = input domain
    % dx        = grid size
    % dt        = time interval
    % N         = total number of equations
    % nx        = total number of grids
    % g         = gravity
    % matrix    = user input either HSWME or BHSWME

    % Output
    % A         = transport matrix
    % D_edge_minus_nx = right boundary
    % D_edge_minus_1  = left boundary

    if (test_case == "Smooth_periodic" )
        w_edge       = roe_avg(w,x);
    
        if (matrix == "HSWME")
            % Hyperbolic shallow water moment model
            A       = HSWME_matrix(nx,N,g,w_edge);       
        else
            % Beta-Hyperbolic shallow water moment model
            A       = H_beta_SWME_matrix(nx,N,g,w_edge); 
        end
        
        % flux calculated at the boundaries
        D_edge_minus_nx = (0.5* (A(:,:,1) - (dx/dt).*eye(N))) * (w(:,1)- w(:,end));
        D_edge_plus_1   = (0.5* (A(:,:,1) + (dx/dt).*eye(N))) * (w(:,1)- w(:,end));
    
        % flux calculated at the boundaries  (Force_scheme)
        % D_edge_minus_nx = (0.5* (A(:,:,1) - (0.5*(dx/dt)*eye(N) + 0.5*(dt/dx)*A(:,:,1).^2  ))) * (w(:,1)- w(:,end));
        % D_edge_plus_1   = (0.5* (A(:,:,1) + (0.5*(dx/dt)*eye(N) + 0.5*(dt/dx)*A(:,:,1).^2  ))) * (w(:,1)- w(:,end));
    
    elseif (test_case == "Dam-break" )
        w_edge       = roe_avg_non_periodic_BC(w);
    
        if (matrix == "HSWME")
            % Hyperbolic shallow water moment model
            A       = HSWME_matrix(nx,N,g,w_edge);       
        else
            % Beta-Hyperbolic shallow water moment model
            A       = H_beta_SWME_matrix(nx,N,g,w_edge); 
        end
    
        % flux calculated at the boundaries
        D_edge_minus_nx  = (0.5* (A(:,:,1)- (dx/dt).*eye(N))) * (w(:,end)- w(:,end));
        D_edge_plus_1    = (0.5* (A(:,:,1) + (dx/dt).*eye(N)))* (w(:,1)- w(:,1));
    else
        return
    end
end