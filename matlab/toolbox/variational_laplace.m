function [Ep,Cp,Eh,Ch,F] = variational_laplace(M,U,Y)
    % Variational Laplace scheme. 
    %
    % Fits any model to data and returns a posterior density over the
    % parameters N(Ep,Cp), a posterior density over the hyperparameters
    % governing the noise N(Eh,Ch), and an approximation of the log model
    % evidence, the Free Energy or Evidence Lower Bound (ELBO) F.
    %
    % Static models have the form:
    %
    % y = g(p,M,U) + e
    % e ~ N(0,Pi)
    % 
    % Where y is the data, g is the generative model, p are the parameters,
    % M is metadata relating to the model, U are the inputs to the model.
    %
    % Dynamic models have the form:
    %
    % df/dx = f(x,u,P,M)
    % y     = g(integral(f)) + e
    %
    % Where x are hidden or latent variables and the integral is over time. 
    %
    % The observation noise is parameterized according to:
    %
    % Pi = exp(h[1])*Q{1} + exp(h(2))*Q{2} + ...
    %
    % Where Q{1...Nh} are precision matrices and h[1...Nh] are the log of
    % hyperparameters governing the precision of the noise.
    %
    % Adapted from spm_nlsi_gn.m by Karl Friston et al. to operate
    % without dependencies on the SPM software package. Note that the
    % original implementation of the scheme has some advantages over the
    % one here, including increased performance by operating with sparse 
    % matrices.
    %
    % Inputs:
    % M.Nmax - maximum iterations (default 100)      [1 x 1]    
    % M.pC   - prior covariance of parameters        [Np x Np]
    % M.pE   - prior expecation of parameters        [Np x 1]
    % M.hC   - prior covariance of hyperparameters   [Nh x Nh]    
    % M.hE   - prior expectation of hyperparameters  [Nh x 1]
    % M.g    - generative model, or...               [handle]
    % M.IS   - integration scheme, in which case:    [handle]
    %          M.f - function that returns df/dx     [handle]
    %          M.g - observation function y=g(x)     [handle]    
    % U      - inputs for the model                  [Nu x Nc]
    % Y.y    - data                                  [Ny x 1]
    % Y.Q    - data precision component(s)           {[Ny x Ny]}
    %
    % Returns:
    % Ep   - posterior expected values of parameters      [Np x 1]
    % Cp   - posterior covariance of parameters           [Np x Np]
    % Eh   - posterior expected values of hyperparameters [Nh x 1]
    % Ch   - posterior covariance of hyperparameters      [Nh x Nh]
    % F    - free energy                                  [1 x 1]
    %
    % _____________________________________________________________________
    % Copyright (C) 2022 Zeidman, Friston, Parr
    % 
    % This program is free software: you can redistribute it and/or modify
    % it under the terms of the GNU General Public License as published by
    % the Free Software Foundation, either version 3 of the License, or
    % (at your option) any later version.
    % 
    % This program is distributed in the hope that it will be useful,
    % but WITHOUT ANY WARRANTY; without even the implied warranty of
    % MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    % GNU General Public License for more details.
    % 
    % You should have received a copy of the GNU General Public License
    % along with this program.  If not, see <https://www.gnu.org/licenses/>.
    % _____________________________________________________________________
            
    % Unpack data    
    y = Y.y;
    if isfield(Y,'Q')
        Q = Y.Q;
    else
        Q = {eye(length(y))};
    end
    
    % Complement model structure with useful quantities
    M.ipC = invs(M.pC);      % Prior precision (parameters)
    M.ihC = invs(M.hC);      % Prior precision (hyperparameters)
    M.Ny  = length(y);       % Number of observations
    M.Np  = size(M.pC,1);    % Number of parameters
    M.Nh  = length(Q);       % Number of hyperparameters
    
    % Starting value for parameters and hyperparameters
    p = M.pE;
    h = M.hE;
    
    % Generative function to use
    if isfield(M,'IS') && ~isempty(M.IS)
        M.fun = M.IS;
    elseif isfield(M,'g') && ~isempty(M.g)
        M.fun = M.g;
    else
        error('Please provide an integrator (M.IS) or a model (M.g)');
    end
    
    % Validate dimensions of inputs
    assert(iscell(Q) && length(Q)==M.Nh && length(h)==M.Nh,...
        'There should be one precision component Q per hyperparameter h');
    assert(size(Q{1},1) == M.Ny),...
        'Please check the size of the precision component(s)';
    assert(size(M.ihC,1) == M.Nh && length(h) == M.Nh,...
        'Check the size of the priors on the hyperparameters');
    assert(size(M.ipC,1) == M.Np && length(p) == M.Np,...
        'Check the size of the priors on the parameters');    
    
    % VL scheme starts here
    % ---------------------------------------------------------------------
    % Inital regularization parameter
    logv = -4;
    
    % Most recently accepted values
    C    = struct();    
    C.p  = [];         % Current estimate of the parameters
    C.h  = [];         % Current estimate of the hyperparameters
    C.F  = -inf;       % Current free energy
    C.Cp = [];         % Current posterior covariance over parameters
    
    dFdp  = [];
    dFdpp = [];
    
    criterion = [0 0 0 0];
    
    % Maximum iterations
    try max_it = M.Nmax; catch, max_it = 100; end

    for it = 1:max_it
               
        % Compute Jacobian dg/dp
        % -----------------------------------------------------------------
        dgdp = compute_g_gradient(p);
                        
        % If needed, increase regularization and re-compute Jacobian
        revert = check_stability(dgdp);
        if revert && it > 1
            [dgdp, p, logv, revert] = repair_gradient(dFdpp,dFdp,C.p,logv);
        end
        
        % Jacobian
        J = -dgdp;
        
        % If gradient is still not stable, give up
        if revert, error('Could not compute gradient'); end
              
        % Optimize hyperparameters
        % -----------------------------------------------------------------
        for it2 = 1:8
            [iS,P] = compute_data_precision(h,Q);      
            [~,Cp] = compute_F_curvature(J,iS);
            
            [h,Ch,has_converged] = update_hyperparameters(h,p,J,Cp,P,iS,y);
            
            if has_converged
                break
            end
        end
        
        % Calcuate free energy and adjust regularization
        % -----------------------------------------------------------------
        F = calc_free_energy(p,h,y,iS,Cp,Ch);              
        
        if F > C.F || it < 3
            % Free energy has improved
            
            % Store current estimates
            C.p = p;
            C.h = h;
            C.F = F;
            C.Cp = Cp;
            
            % Compute free energy gradient and curvature
            dFdp  = compute_F_gradient(p,y,J,iS,Ch,P);
            dFdpp = compute_F_curvature(J,iS);
            
            % Decrease regularization
            logv = decrease_regularization(logv);
            
            fprintf('VL:(+) ');
        else
            % Free energy has got worse
            
            % Restore previously accepted estimates
            p  = C.p;
            h  = C.h;
            Cp = C.Cp;
            
            % Increase regularization
            logv = increase_regularization(logv);
            
            fprintf('VL:(-) ');
        end        
        
        % Report
        % -----------------------------------------------------------------
        fprintf('It %d: log(v)=%2.2f, F=%2.2f\n',it,logv,C.F);
        
        % Optimize parameters
        % -----------------------------------------------------------------
        dp = update(dFdpp, dFdp, logv);
        p  = p + dp;
        
        % Convergence check
        % -----------------------------------------------------------------
        dF = dFdp' * dp;
        criterion = [(dF < 1e-1) criterion(1:end - 1)];
        if all(criterion)
            disp('Convergence');
            break;
        end        
    end
    
    % Outputs
    Ep = C.p;
    Cp = C.Cp;   
    Eh = C.h;    
    F  = C.F;
    
% -------------------------------------------------------------------------
function dgdp = compute_g_gradient(p)
    % Numerical estimate of gradient dg/dp using finite differences
    %
    % p    - parameters [Np x 1]
    % dgdp - dg/dp      [Np x 1]
    
    dgdp = zeros(M.Ny,M.Np);    
    dx = exp(-8);
    p0 = p;
    for i = 1:M.Np
        p = p0;
        p(i) = p(i) + dx;
        dgdp(:,i) = (M.fun(p,M,U) - M.fun(p0,M,U)) ./ dx;
    end
end
% -------------------------------------------------------------------------
function dFdp = compute_F_gradient(p,y,J,iS,Ch,P)    
    % Computes gradient of the expected log joint wrt parameters
    % (where the expectation is wrt to the hypeparameters)
    %
    % p    - parameters                              [Np x 1]
    % y    - data                                    [Ny x 1]
    % J    - Jacobian dgdp                           [Ny x Np]
    % iS   - estimated data precision                [Ny x Ny]
    % Ch   - posterior covariance of hyperparameters [Nh x Nh]
    % P    - estimated data precision per component  {Nh x Nh}
    % dFdp - gradient of the free energy             [Np x 1]
    
    % Error terms
    ep = p - M.pE;
    ey = y - M.fun(p,M,U);
            
    % Gradient of the expected log joint 
    dFdp  = -J'*iS*ey - M.ipC*ep;
    
    % Optional term
    %for i = 1:length(P)
    %    dFdp = dFdp - 0.5*Ch(i,i)*J'*P{i}*ey;
    %end
end
% -------------------------------------------------------------------------
function [dFdpp,Cp] = compute_F_curvature(J,iS)    
    % Computes 2nd derivative of the expected log joint wrt parameters
    % (where the expectation is wrt to the hypeparameters)
    %
    % J     - Jacobian dgdp                           [Ny x Np]
    % iS    - estimated data precision                [Ny x Ny]
    % dFdpp - second derivative (Hessian)            [Np x Np]
    % Cp    - posterior covariance                   [Np x Np]
    
    dFdpp = -J'*iS*J - M.ipC;
    
    % Posterior precision (Pp) and covariance (Cp) of parameters
    Pp = -dFdpp;
    Cp = invs(Pp);        
end
% ---------------------------------------------------------------------
function [dgdp, p, logv, revert] = repair_gradient(dFdpp,dFdp,p,logv)
    % Attempts to recover from a failed attempt to compute the gradient of
    % the generative model by increasing regularization
    %
    % dFdpp  - second derivative (Hessian)             [Np x Np]
    % dFdp   - gradient of the free energy             [Np x 1]
    % p      - most recently accepted parameters       [Np x 1]
    % logv   - regularization parameter                [1 x 1]
    % revert - binary flag indicating failure          [1 x 1]
    
    p_ok = p;
    for i = 1:4
        % Increase regularization by decreasing v
        logv = increase_regularization(logv);

        % Update parameters
        p = p_ok + update(dFdpp, dFdp, logv);

        % Try again
        dgdp   = compute_g_gradient(p,M,U);
        revert = check_stability(dgdp);

        % Stop if OK now
        if ~revert, break, end
    end
       
end
% ---------------------------------------------------------------------
function is_unstable = check_stability(dFdp)       
    % Tests whether the gradient dFdp is well behaved
    % dFdp   - gradient of the free energy [Np x 1]    

    normdfdp     = norm(dFdp,'inf'); % Maximum absolute row sum
    is_unstable  = isnan(normdfdp) || normdfdp > 1e32;
end
% -------------------------------------------------------------------------
function logv = increase_regularization(logv)
    % Increase regularization by decreasing log(v)
    logv = min(logv - 2,-4);
end

% -------------------------------------------------------------------------
function logv = decrease_regularization(logv)
    % Decrease regularization by increasing log(v)
    logv = min(logv + 1/2,4);
end

% -------------------------------------------------------------------------
function dp = update(dFdpp, dFdp, logv) 
    % Calculates update on the parameters using local linearization (Ozaki
    % 1985). Adapted from spm_dx by Karl Friston
    %
    % dFdp   - gradient of the free energy             [Np x 1]
    % dFdpp  - second derivative (Hessian)             [Np x Np]
    % logv   - regularization parameter                [1 x 1]
    % dp     - change in parameters                    [Np x 1]
    
    n = length(dFdp);
    
    %TODO comment this
    t = exp(logv - logdet(dFdpp)/n);
    
    %Compute update using local linearization
    dp = (expm(dFdpp*t) - eye(n))*inv(dFdpp)*dFdp;
end

% -------------------------------------------------------------------------
function F = calc_free_energy(p,h,y,iS,Cp,Ch)
    % Calculate the free energy
    %
    % p  - parameters                               [Np x 1]
    % h  - hyperparameters                          [Nh x 1]
    % y  - data                                     [Ny x 1]
    % iS - data precision                           [Ny x Ny]
    % Cp - posterior covariance (parameters)        [Np x Np]
    % Ch - posterior covariance (hyperparameters)   [Nh x Nh]
    
    % Error terms
    ey = y - M.fun(p,M,U);
    ep = p - M.pE;  
    eh = h - M.hE;
    
    % Prior covariances
    pC = M.pC;
    hC = M.hC;
    
    % Prior precisions
    ipC = M.ipC;
    ihC = M.ihC;
    
    % Data covariance    
    S = invs(iS);
    
    % Number of observations
    Ny = M.Ny;
    
    L(1) = logdet(S)  + (ey'*iS*ey);    % Likelihood
    L(2) = logdet(pC) + (ep'*ipC*ep);   % Prior (parameters)
    L(3) = logdet(hC) + (eh'*ihC*eh);   % Prior (hyperparameters)
    L(4) = logdet(Cp) + logdet(Ch);     % Posterior entropy
    L(5) = Ny*log(2*pi);                 % Constants
    
    F = -0.5*L(1) -0.5*L(2) -0.5*L(3) +0.5*L(4) -0.5*L(5);                
end

% -------------------------------------------------------------------------
function [iS, P] = compute_data_precision(h,Q)
    % Calculates estimated precision of the data
    % 
    % h - hypeparameters       [Nh x 1]
    % Q - precision components {Ny x Ny}
    
    Ny = M.Ny;
    Nh = M.Nh;
    
    iS = zeros(Ny,1);
    P  = cell(Nh,1);
    for i = 1:length(Q)
        P{i} = Q{i}*(exp(-32) + exp(h(i)));
        iS = iS + P{i};
    end
end
% -------------------------------------------------------------------------
function [h,Ch,has_converged] = update_hyperparameters(h,p,J,Cp,P,iS,y)
    % Calculates update on the hyperparameters
    % h    - old hyperparameters                                [Nh x 1]
    % p    - parameters                                         [Np x 1]
    % J    - Jacobian dgdp                                      [Ny x Np]
    % Cp   - parameter covariance                               [Np x Np]
    % P{i} - contribution to data precision from i-th component [Ny x Ny]
    % iS   - data precision                                     [Ny x Ny]
    % y    - data                                               [Ny x 1]
    %
    % Returns:
    % h             - new hyperparameters                       [Nh x 1]
    % Ch            - hyperparameter covariance                 [Nh x Nh]
    % has_converged - binary flag for convergence               [1 x 1]
    
    % Re-compute error terms
    ey = y - M.fun(p,M,U);       
    eh = h - M.hE;
    
    % Prior precision of hyperparameters
    ihC = M.ihC;
    
    % Estimated data covariance
    S = invs(iS);
    
    % Estimate second derivatives of F (approximate Hessian)
    Nh = M.Nh;
    dFdhh = zeros(Nh,Nh);
    for i = 1:Nh
        dFdhh(i,i) = -ihC(i,i) ...
                     + 0.5 * trace(P{i}*S - P{i}*S*P{i}*S)...
                     - 0.5 * (ey' * P{i} * ey) ...
                     - 0.5 * trace(Cp * J' * P{i} * J);
    end
    
    % Posterior precision (Ph) and covariance (Ch) of the hyperparameters
    Ph = -dFdhh;
    Ch = invs(Ph);            

    % Gradient of the expected log joint wrt each hyperparameter
    dFdh = zeros(Nh,1);
    for i = 1:Nh

        % d_eh / d_lambda{i}
        deh = zeros(Nh,1);
        deh(i) = 1;

        dFdh(i) = 0.5 * trace(P{i} * S) ...
                 -0.5 * (ey' * P{i} * ey) ...
                 -0.5 * trace(Cp * J' * P{i} * J) ...
                 -deh' * ihC * eh;
    end            

    % Update hyperparameters with low regularization     
    dh = update(dFdhh, dFdh, 4);
    
    % Bound step size between -1 and 1
    dh = max(dh, -1);
    dh = min(dh, 1);
    
    % Update
    h = h + dh;
    
    % Check for convergence
    has_converged = (dFdh'*dh) < 1e-2;
end

% -------------------------------------------------------------------------
function H = logdet(C)
    % H = log(det(C))
    % Adapted from spm_logdet by Karl Friston and Ged Ridgway
    
    % Remove any rows / columns with zero on the leading diagonal
    i = find(diag(C));
    C = C(i,i);
    
    % Get all non-zero entries
    [~,~,s] = find(C);
    
    if any(isnan(s)), H = nan; return; end
    
    % Non-diagonal matrices
    if ~isdiag(C)
        if issymmetric(C)
            % Symmetric non-diagonal matrix
            try
                % Try positive definite (Cholesky factorization)
                R = chol(C);
                H = 2*sum(log(diag(R)));
                return
            catch
                % Not positive definite
                s = svd(C);
            end            
        else
            % Asymmetric non-diagonal matrix
            s = svd(full(C));
        end
    end
    
    % C is either a diagonal matrix or a full asymmetric matrix
    % Multiply the leading diagonal in log space to avoid numerical issues
    TOL = 1e-16;
    s   = s(s > TOL & s < 1/TOL);
    H   = sum(log(s));
end
% -------------------------------------------------------------------------
function X = invs(A)
    % Safe invert - adapted from spm_inv by Karl Friston
    X = inv(A + eye(size(A))*exp(-32));
end

end