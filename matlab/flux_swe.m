function [D_edge_plus,D_edge_minus] = flux_swe (nx,N,x,dx,dt,g,w,test_case,matrix)
    % Input variables
    % nx        = total number of grids
    % N         = total number of equations
    % x         = input domain
    % dx        = grid size
    % dt        = time interval
    % g         = gravity
    % w         = initial variables h, u and alphas
    % test case = user input either Smooth_periodic or Dam-break
    % matrix    = user input either HSWME or BHSWME

    % Output 
    % D_edge_plus   = flux calculated at right cell
    % D_edge_minus  = flux calculated at left cell
  D_edge_minus = zeros(N,length(x));
  D_edge_plus  = zeros(N,length(x));

  [A,D_edge_minus_nx,  D_edge_plus_1] = boundary_condition(test_case,w,x,dx,dt,N,nx,g,matrix);
    
    % flux calculated at the interfaces
     for i = 1:nx-1
         D_edge_minus(:,i)  = (0.5* (A(:,:,i+1) - (dx/dt)*eye(N)))* (w(:,i+1)-  w(:,i));
    
     end
    
     for i = 2:nx
         D_edge_plus(:,i)  = (0.5* (A(:,:,i) + (dx/dt)*eye(N)))* (w(:,i)- w(:,i-1));
    
     end

    % flux calculated at the interfaces (Force_Scheme)
    %  for i = 1:nx-1
    %      D_edge_minus(:,i)  = (0.5* (A(:,:,i+1) -  (0.5*(dx/dt)*eye(N) + 0.5*(dt/dx)*A(:,:,i+1).^2  )     ))* (w(:,i+1)-  w(:,i));
    % 
    %  end
    % 
    %  for i = 2:nx
    %      D_edge_plus(:,i)  = (0.5* (A(:,:,i) + (0.5*(dx/dt)*eye(N) + 0.5*(dt/dx)*A(:,:,i).^2  )))* (w(:,i)- w(:,i-1));
    % 
    %  end

    % Assign the values at the left and right boundary
    D_edge_minus(:,nx)  = D_edge_minus_nx;          % right boundary
    D_edge_plus(:,1)    = D_edge_plus_1;            % left boundary
end
