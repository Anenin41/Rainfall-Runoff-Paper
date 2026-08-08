%**************************************************************************
%           1d Hyperbolic Shallow Water Moment Model (HSWME) 
%              Enter the user input to compute the system
%                     dw/dt + A (dw/dx) = S_w 
%               where w = (h,hu,halpha1,.....,halphaN)
%
%                   Coded by Afroja Parvin on 05.07.2023 
%**************************************************************************

% Function: input takes all the user input, initial grid, and parameters needed for the model.
%
% Function: initial conditions return all the initial values for w.
%
% Function: HSWME or H_beta_HSWME returns the system/transport matrix.
%
% Function: flux_swe calculates flux at all cells; calls boundary function inside.
%
% Function: friction computes source term on the right-hand side of the system.
%
% Function: roe_avg (for periodic boundary) or roe_avg_non_periodic_BC (for non_pperiodic boundary)
% scheme for sapce discretization.
%
% Function: RK34_SWE used for the time integration.
%
% Function: plot_var plots h or u or alpha(s). Depending on the number of
% moments as user input, alpha(s) figures will be generated.

clc; clear all;

disp("Please enter the following informations to run the model")
Question_1   = input("Please enter the number of moments n= ");
n            = Question_1;
Question_2   = input("Which test case do you want to consider (Smooth_periodic or Dam-break)? ","s");
test         = Question_2;
Question_3   = input("Which transport matrix A do you want to consider (HSWME or BHWME)? ","s");
matrix       = Question_3;
disp("Thank you! Please wait for the result ...")

% Call all the input variables
in = input_swe(n,test,matrix);

% Assign initial condition
W  = initial_condition (in.x, in.N, in.n, in.u, in.h);
W0 = W;
x  = in.x;

% time count start
tic  
%% Time integration Scheme
while(in.t<in.Tmax)
    [W, in.t]  = RK34_SWE(W, in.dt, in.t, @non_lin_trend, in.NX,in.N,in.x, in.dx,in.g,in.nu,in.slip_length,in.test_case,in.matrix);
end
% time count end
toc  

%% Plot the variable you want 
%  plot water height
figure(1)   
    plot(x,W(1,:),'b','LineWidth',2)
    title(sprintf('Water height at t = %0.2f',in.t))
    xlabel('x');
    ylabel('h');
% plot mean velocity
figure(2)   
    plot(x,W(2,:)./W(1,:),'r','LineWidth',2)
    title(sprintf('Water velocity at t = %0.2f',in.t))
    xlabel('x');
    ylabel('u_m');

%plot all the moments
for i = 1:n
    fig_num = i + 2;  % Avoids conflict with figure(2)
    figure(fig_num)
    plot(x,W(i+2,:)./W(1,:),'k','LineWidth',2)
    xlabel('x');
    ylabel(sprintf("$\\alpha_{%3d}$",i),'Interpreter','latex')
    title(sprintf('moments at t = %0.2f',in.t))
end
