function [y,ty,x,tx] = ode_LL(P,M,U)
    % Performs integration using Local Linearization for a system of the
    % form:
    %
    % df/dx = f(x,u,P,M)
    % y     = g(integral(f)) + e
    %
    % Inputs:
    % P     - model parameters                          [Np x 1]
    % M.x   - starting values for hidden states         [Nx x 1]
    % M.ns  - number of samples                         [1 x 1]
    % M.f   - model returning df/dx                     [handle]
    % M.g   - observation model                         [handle]
    % U.u   - inputs for the model with Nu time bins    [Nu x Nc]
    % U.dt  - length of one time bin in seconds         [1 x 1]
    %
    % Returns:
    % y     - predicted data                            [Ns x 1]
    % ty    - time of each sample in seconds            [Ns x 1]
    % x     - predicted latent variables                [Nu x Nx]
    % tx    - time for latent variables                 [Nu x 1]
    %
    % See:
    % Ozaki, T. 1985. Non-linear time series models and dynamical systems. 
    %
    % Zeidman, Friston, Parr
    % _____________________________________________________________________

    nx = length(M.x); % Number of hidden states
    ns = M.ns;        % Number of samples
    nu = size(U.u,1); % Number of integration time bins
    
    % Compute Jacobian df/dx at t0
    dfdx = compute_gradient(P,M,U.u(1,:));

    % Evaluate Q = (expm(dt*J) - I)*inv(J)    
    dfdx  = dfdx - eye(nx,nx)*exp(-16);
    Q     = (expm(U.dt*dfdx) - eye(nx,nx))/dfdx;

    % Integrate
    x = zeros(nx,nu);
    x(:,1) = M.x;    
    for i = 2:nu
        u = U.u(i,:);
        x(:,i) = x(:,i-1) + Q * M.f(x(:,i-1),u,P,M);
    end    
    
    % Integration times
    tx = (0:(nu-1)) * U.dt;
    
    % Bin numbers (yidx) for each sample         
    bins_per_y = nu/ns;
    yidx = ceil((0 : ns-1)*bins_per_y) + 1;
    
    % Shift bin numbers to align the start of each measurement to
    % half way though the acquisition time (microtime onset)
    offset_bins = floor(bins_per_y/2);
    yidx = yidx + offset_bins;    
    
    % Acquisition time for each sample
    ty = (yidx-1) * U.dt;
    
    % output - implement g(x)
    y = zeros(ns,1);
    for i = 1:ns
        bin = yidx(i);
        y(i,:) = M.g(x(:,bin),U.u(bin,:),P,M)';
    end
        
    x = x';    
end

function dfdx = compute_gradient(P,M,u)
    % Numerical estimate of gradient df/dx using finite differences       
    nx   = length(M.x);
    dfdx = zeros(nx,nx);    
    dx   = exp(-8);
    x0   = M.x;
    for i = 1:nx
        x = x0;
        x(i) = x(i) + dx;
        dfdx(:,i) = (M.f(x,u,P,M) - M.f(x0,u,P,M)) ./ dx;
    end
end