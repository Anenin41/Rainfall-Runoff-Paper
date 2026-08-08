function A = H_beta_SWME_matrix(nx,N,g,w)   % Beta-Hyperbolic shallow water moment model
    % Input variables
    % nx        = total number of grids
    % N         = total number of equations
    % g         = gravity
    % w         = initial variables h, u and alphas
    
    % Output 
    % A         = returns system matrix
n = N-2;
A = zeros (N,N,nx);
    
    
    if (n==1)                  % when the number of moment is one
    
        for i = 1:nx
            A(1,2,i) =   1; 
            A(2,1,i) =  g*w(1,i)-(w(2,i)./w(1,i)).^2  -(1/3)* (w(3,i)./w(1,i)).^2;
            A(2,3,i) =  (2/3)* (w(3,i))./(w(1,i));
            A(3,1,i) = -2*(w(2,i).*w(3,i))./(w(1,i)).^2;
            A(3,2,i) =  2*(w(3,i)./w(1,i));
            A(2,2,i) =  2*(w(2,i))./(w(1,i));
            A(3,3,i) =  (w(2,i))./(w(1,i));
        end  
    
    
    elseif(n>=2)               % when the number of moment is greater than two
        
        for i= 2:n
            c_A(i)   = (i+1)./(2*i+1);
        end
    
        for i= 2:n-1
            a_A(i)   = (i-1)./(2*i-1);
        end
        a_A(n)= (2*n.^2-n-1)/(2*n.^2+n-1);
    
       for j = 3:N-1
        for i = 1:nx
    
              A(j,j,i)   = (w(2,i))./(w(1,i));
              A(j+1,j,i) = a_A(j-1).*(w(3,i))./(w(1,i));
              A(j,j+1,i) = c_A(j-1).*(w(3,i))./(w(1,i)); 
        end  
       end
      for i = 1:nx
              A(1,2,i) = 1;
              A(2,1,i) = g*w(1,i)-(w(2,i)./w(1,i)).^2  -(1/3)* (w(3,i)./w(1,i)).^2;
              A(2,3,i) = (2/3)* (w(3,i))./(w(1,i));
              A(3,1,i) = -2*(w(2,i).*w(3,i))./(w(1,i)).^2;
              A(3,2,i) = 2 * (w(3,i)./w(1,i));
              A(4,1,i) = (-2/3)*((w(3,i))./(w(1,i))).^2;
              A(2,2,i) = 2*(w(2,i))./(w(1,i)); 
              A(N,N,i) = (w(2,i))./(w(1,i));
      end
    
    else 
         for i = 1:nx               % 1D Shallow water Model
            A(1,2,i) =   1; 
            A(2,1,i) =  g*w(1,i)-(w(2,i)./w(1,i)).^2;
            A(N,N,i) =  2*(w(2,i))./(w(1,i));
        end  
    end

end





























% A = zeros (N,N,nx);
% d = zeros(1,4);
% 
% %for j = 2:3
% for i = 1:nx
%     d(1) = g*w(1,i)-(w(2,i)./w(1,i)).^2  -(1/3)* (w(3,i)./w(1,i)).^2;
%     d(2) = (2/3)* (w(3,i))./(w(1,i));
%     d(3) = -2*(w(2,i).*w(3,i))./(w(1,i)).^2;
%     d(4) =  2 * (w(3,i)./w(1,i));
%     A(1,2,i) = 1; 
%     A(2,1,i) = d(1);
%     A(2,3,i) = d(2);
%     A(3,1,i) = d(3);
%     A(3,2,i) = d(4);
%     A(2,2,i) = 2*(w(2,i))./(w(1,i));
%     A(3,3,i) = (w(2,i))./(w(1,i));

% %UNTITLED Summary of this function goes here
% syms a g h u;
% A = zeros (N,N,nx);
% m = A_matrix(N);
% for i = 1:nx
% A(:,:,i) = subs(m, {g, h, u, a}, {1, w(1,i), w(2,i)./w(1,i), w(3,i)./w(1,i)});
% end

