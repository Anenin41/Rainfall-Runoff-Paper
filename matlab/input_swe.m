function input = input_swe(n,test,matrix)

% Input           
% user input from the main file

% length scale (m)
ax                = [-1,1];       % domain
input.n           =  n;           % number of moment
input.NX          =  800;        % number of grid cells
input.N           =  input.n + 2; % number of equation

% resolution
input.dx          = diff(ax)/input.NX;
input.x           = ax(1)+input.dx/2:input.dx:ax(2)-input.dx/2; % cell centers

% Physical constants
input.g           = 1.0;
input.nu          = 0.1;
input.slip_length = 0.1;                    % slip_length= 0.0001 for smoothing
input.cfl         = 0.7;                    % CFL constant

% Test Case you want to consider
input.test_case   = test;

% The matrix A you want to consider
input.matrix      = matrix;

% Time to compute the test case
    if (input.test_case == "Smooth_periodic" )
        input.Tmin        = 0;
        input.Tmax        = 2;              
        input.t           = input.Tmin;
        input.dt          = 0.001;
        input.dt_save     = 0.5;
    elseif(input.test_case == "Dam-break" )
        input.Tmin        = 0;
        input.Tmax        = 0.2;         
        input.t           = input.Tmin;
        input.dt          = 0.001;
        input.dt_save     = 0.05;
    else
        return
    end

 % Initial height according to the test case
    if (input.test_case == "Smooth_periodic" )
        input.h  = 1 + exp(3*cos(pi*(input.x+ 0.5)))./exp(4);
    elseif (input.test_case == "Dam-break" )
        input.h  = ones(size(input.x));
        input.h(input.x<=0)= 1.5;
    else
        return
    end
    
  % Initial velocity function
    input.u           = @ (z) 0.50*z;
end

