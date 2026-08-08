function trend = non_lin_trend(nx,N,x,dx,dt,g, nu,slip_length,w,test_case,matrix)
% Function that computes the system
% S_w           = source term (friction)
% D_plus, D_min = return the flux at the entire domain

     % return friction at the cell center
     S_w             = friction(nx,N,nu,slip_length,w);
    
     % return the flux at the entire domain
     [D_plus, D_min] = flux_swe (nx,N,x,dx,dt,g,w,test_case,matrix);
    
     % compute dw/dt
     trend           =   -(1/dx)*((D_plus + D_min)) + S_w;
end