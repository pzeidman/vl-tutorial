function [y,ty,x,tx] = vl_ode_wrapper(P,M,U,integrator)
    % Wrapper to enable SPM or Matlab integrators to be used with
    % variational_laplace.m.
    % 
    % P     - model parameters                          [Np x 1]
    % M.x   - starting values for hidden states         [Nx x 1]    
    % M.ns  - number of samples                         [1 x 1]
    % M.f   - model returning df/dx                     [handle]
    % M.g   - observation model                         [handle]    
    % U.u   - inputs for the model with Nu time bins    [Nu x Nc]
    % U.dt  - length of one time bin in seconds         [1 x 1]    
    %
    % integrator - function handle for the desired 
    %              integrator, e.g. @ode113 or @spm_int
    %
    % Returns:
    % y     - predicted data                            [Ns x 1]
    % ty    - time of each sample in seconds            [Ns x 1]
    % x     - predicted latent variables                [Nu x Nx]
    % tx    - time for latent variables                 [Nu x 1]
    % 
    % NB:  only y is returned if using SPM integrators. 
    % NB2: if using SPM integrators, make sure to specify
    % M.m (number of inputs), M.n (number of states), M.l (number of
    % outputs)
    %
    % Zeidman, Friston, Parr
    % _____________________________________________________________________
    
    if nargin < 4, integrator = @spm_int; end
    
    % Number of integration time bins (nu) and samples (ns)
    nu = size(U.u,1);
    try ns = M.ns; catch, ns = nu; end
    
    % Duration of each measurement
    bins_per_sample = nu/ns;     
    secs_per_sample = bins_per_sample * U.dt;
    
    ty = [];
    x  = [];
    tx = [];
    
    if startsWith(char(integrator), 'spm_')
        % Use SPM's integrator            
        y = spm_int(P,M,U); 
    else
        % Use one of the Matlab integrators
        % Adapted from spm_int_ode.m by Karl Friston
        
        try U.dt; catch, U.dt = 1; end

        % Onset times of integration time bins
        tspan = (1:nu)*U.dt;            

        % Get functions and initial states
        f = M.f;
        g = M.g;
        x = M.x;

        % Integrate hidden states
        ode     = @(t,x) f(x,U.u(ceil(t/U.dt),:),P,M);           
        options = odeset('RelTol',1e-2,'AbsTol',1e-4);
        [tx,x]  = integrator(ode,tspan,x,options);

        % Bin numbers (by) for each sample                   
        by = ceil((0 : ns-1)*bins_per_sample) + 1;

        % Shift bin numbers to align the start of each measurement to
        % half way though the acquisition time (microtime onset)
        microtime_offset_bins = floor(bins_per_sample/2);
        by = by + microtime_offset_bins;

        % Acquisition time for each sample            
        microtime_offset_secs = microtime_offset_bins * U.dt;
        ty = (0:ns-1) * secs_per_sample + microtime_offset_secs;

        % output - implement g(x)
        y = zeros(ns,1);
        for i = 1:ns
            bin = by(i);
            y(i,:) = g(x(bin,:),U.u(bin,:),P,M)';
        end
    end
    
end