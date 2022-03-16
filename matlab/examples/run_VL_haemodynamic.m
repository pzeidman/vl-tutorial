function run_VL_haemodynamic
% VL scheme - DCM haemodynamic model example
%
% Hidden states (x):
% 1. S    - vasoactive signal
% 2. logf - log rCBF
% 3. logv - log venous blood volume
% 4. logq - log dHb
%
% Parameters (P):
% 1. log_transit - haemodynamic transit rate (Hz)
% 2. log_decay   - decay rate of vasoactive signal (Hz)
% 3. efficacy    - stimulus efficacy (Hz)
%
% To read about the model, please see:
%
% Friston, K. J., et al. (2000). "Nonlinear responses in fMRI: the 
% Balloon model, Volterra kernels, and other hemodynamics." 
% Neuroimage 12(4): 466-477.
%	
% Stephan, K. E., et al., 2007. Comparing hemodynamic models with DCM. 
% Neuroimage, 38, 387-401.
% 
% Zeidman, P. et al. 2019a. A guide to group effective connectivity 
% analysis, part 1: First level analysis with DCM for fMRI. NeuroImage, 200, 
% 174-190.
% 
% Zeidman, Friston, Parr
% _________________________________________________________________________
    
% define model
% -------------------------------------------------------------------------
M   = struct();
M.f = @fx;         % model returning df/dx (dynamics of hidden states)
M.g = @gx;         % model returning y = gx(x) (BOLD response)
M.x  = zeros(4,1); % initial states

% define timing
% -------------------------------------------------------------------------
TR   = 3;                % Time of one MRI volume (secs)
ns   = 100;              % Number of samples (MRI volumes)
dt   = 1/16;             % Length of one time bin for integration (secs)
nu   = ceil(ns*TR*1/dt); % Number of integration time bins

% Input structure
U.u  = zeros(nu,1);
U.dt = dt;

% Compute timing for stimulus onsets, evenly spaced +- 15s jitter for
% sampling efficiency
ntrials = 10;
jitter  = 1+(15-1).*rand(1,ntrials);
ons = linspace(TR,TR*ns-30,ntrials) + jitter;
dur = 1;
for i=1:ntrials
    t_start  = round(ons(i)*1/dt) + 1;
    t_end    = t_start + round(dur*1/dt) - 1;
    U.u(t_start:t_end) = 1;
end

% define integrator
% -------------------------------------------------------------------------
M.IS = @ode_LL;
M.ns = ns;

% set priors
% -------------------------------------------------------------------------
M.pE = [0 0 0]';                   % prior expectation
M.pC = diag([1/16,1/16,1]);        % prior covariance
M.hE = 4;                          % prior expected log noise precision
M.hC = 1/8;                        % prior variance of log noise precision

% simulate data
% -------------------------------------------------------------------------
gp  = [0.2 0.4 1]';          % generative parameters
gh  = 4;                     % generative hyperparameters

% Generate data
% -------------------------------------------------------------------------

% Generate signal
[y,ty,x,tx] = ode_LL(gp,M,U);

% Generative noise precision matrix
P = exp(gh)*eye(ns);

% Add noise
rng(1)
y = y + spm_normrnd(zeros(ns,1),inv(P),1);

% Pack
Y = struct();
Y.y  = y;
Y.dt = TR;

% Plot states and data
figure;
subplot(2,1,1);
plot(tx,x,'.-');title('Simulated states');
subplot(2,1,2);
plot(ty,y,'.-');title('Simulated data');
xlabel('Time (secs)');

% Invert
% -------------------------------------------------------------------------
[Ep,Cp,Eh,Ch,F] = variational_laplace(M,U,Y);

% Invert GLM using SPM NLSI for comparison
% -------------------------------------------------------------------------
M.noprint = true;
[Ep2,Cp2,Eh2,F2,~,~,~,Ch2] = spm_nlsi_GN(M,U,Y);

% Plot parameters,  hyperparameters and model fit
% -------------------------------------------------------------------------
figure;
subplot(2,2,1);
plot_parameters(Ep,Cp,gp);
ylim([0 0.6]);
axis square;
xlabel('Parameter');
title(sprintf('Parameters. F=%2.2f',F));
set(gca,'FontSize',12);
ylim([0 1.2]);

subplot(2,2,2);
plot_parameters(Eh,Ch,gh);
ylim([0 7]);
axis square;
xlabel('Hyperparameter');
title('Hyperparameters');
set(gca,'FontSize',12);

subplot(2,2,3);
plot_parameters(Ep2,Cp2,gp);
ylim([0 0.6]);
axis square;
xlabel('Parameter');
title(sprintf('Parameters - SPM. F=%2.2f',F2));
set(gca,'FontSize',12);
ylim([0 1.2]);

subplot(2,2,4);
plot_parameters(Eh2,Ch2,gh);
ylim([0 7]);
axis square;
xlabel('Hyperparameter');
title('Hyperparameters - SPM');
set(gca,'FontSize',12);

% Plot fit
[y,ty,x,tx] = ode_LL(gp,M,U);
figure;
subplot(2,2,1);
plot(ty,Y.y,'--','Color','k');
hold on
plot(ty,y);
xlabel('Time (secs)'); ylabel('y');
axis square
set(gca,'FontSize',12);

end

% -------------------------------------------------------------------------
function dfdx = fx(x,u,P,M)
	% Haemodynamic model
    %
    % x - vector of states
    % u - vector of inputs at the current time
    % P - vector of parameters
    % M - model metadata
    
    dfdx = zeros(4,1); 
    
    % Indices of states in x
    STATE_S = 1; STATE_LOGF = 2; STATE_LOGV = 3; STATE_LOGQ = 4;    
    
    % Indices of parameters in P
    PARAM_LOGTRANSIT = 1; PARAM_LOGDECAY = 2; PARAM_EFFICACY = 3;
    
    % Other constants
    alpha    = 0.32;
    E0       = 0.4;
    feedback = 0.32;
    
    % Unpack current states and un-log     
    S = x(STATE_S);
    f = exp(x(STATE_LOGF));
    v = exp(x(STATE_LOGV));
    q = exp(x(STATE_LOGQ));
    
    % Unpack parameters, un-log and multiply default values
    transit  = 2.00 * exp(P(PARAM_LOGTRANSIT));
    decay    = 0.64 * exp(P(PARAM_LOGDECAY));
    efficacy = 1.00 * P(PARAM_EFFICACY);
    
    % CBF    
    dfdx(STATE_LOGF) = S;
    dfdx(STATE_S)    = efficacy*u - decay*S - feedback*(f-1);
    
    % Outflow    
    fout  = v ^ (1/alpha);
    
    % Venous balloon
    dfdx(STATE_LOGV) = transit * (f - fout);
    
    % dHb
    dfdx(STATE_LOGQ) = transit * ...
                    (f * ((1 - (1-E0)^(1/f))/E0) - fout*q/v); 
                
    % handle logged variables
    dfdx(STATE_LOGF) = dfdx(STATE_LOGF) / f;
    dfdx(STATE_LOGV) = dfdx(STATE_LOGV) / v;
    dfdx(STATE_LOGQ) = dfdx(STATE_LOGQ) / q;
                
end

function y = gx(x,u,P,M)
	% BOLD signal model
    %
    % x - vector of states
    % u - vector of inputs at the current time
    % P - vector of parameters
    % M - model metadata
    
    % Indices of states in x
    STATE_LOGV = 3; STATE_LOGQ = 4;    
    
    % Other constants
    E0  = 0.4;
    nu0 = 40.3;
    TE  = 0.03;
    r0  = 25;
    V0  = 4;

    % Unpack current states and un-log     
    v = exp(x(STATE_LOGV));
    q = exp(x(STATE_LOGQ));
    
    % Unpack parameters, un-log and multiply default value
    epsilon = exp(0); % 3T
    
    % Observation model
    k1 = 4.3 * nu0 * E0 * TE;
    k2 = epsilon * r0 * E0 * TE;
    k3 = 1 - epsilon;
    y  = V0 * (k1*(1-q) + k2*(1-q/v) + k3*(1-v));
    
end