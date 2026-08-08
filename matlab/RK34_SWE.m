function [w_swe, t1] = RK34_SWE(w_swe, dt, t1, non_lin_trend, nx,N,x, dx, g,nu,slip_len,test_case,matrix)
% Third order four stage strongly stability preserving Runge-Kutta integration scheme

% Input variables:
% dt       = time step
% t        = time (unused)
% trend    = function handle for trend 
% NX       = Number of step in space
% dx       = grid spacing in x
% g        = gravity

alpha = [ 1   0  0  0 ; ...
          0   1  0  0 ; ...
         2/3  0 1/3 0 ; ...
          0   0  0  1 ];

beta = [1/2  0   0    0 ; ...
         0  1/2  0    0 ; ...
         0   0  1/6   0 ; ...
         0   0   0   1/2 ];
U0 = w_swe;
U1 = alpha(1,1)*U0                 + dt * beta(1,1)*non_lin_trend(nx,N,x,dx,dt,g, nu,slip_len,U0,test_case,matrix);
U2 = alpha(2,2)*U1                 + dt * beta(2,2)*non_lin_trend(nx,N,x,dx,dt,g, nu,slip_len,U1,test_case,matrix);
U3 = alpha(3,1)*U0 + alpha(3,3)*U2 + dt * beta(3,3)*non_lin_trend(nx,N,x,dx,dt,g, nu,slip_len,U2,test_case,matrix);
w_swe  = alpha(4,4)*U3             + dt * beta(4,4)*non_lin_trend(nx,N,x,dx,dt,g, nu,slip_len,U3,test_case,matrix);
t1 = t1 + dt;
end