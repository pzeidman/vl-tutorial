%% VL Scheme - Single exponential example

% define model
% -------------------------------------------------------------------------
M   = struct();
f   = @(x,u,P,M) -exp(P) * x;   % Exponential decay of variable x
g   = @(x,u,P,M) x;             % Observation model (x is fully observed)
M.f = f;
M.g = g;
M.x  = 1;                       % initial state of x

% define timing
% -------------------------------------------------------------------------
ny = 100; % Number of samples
M.ns = ny;

% Inputs
U.u  = zeros(ny,1);
U.dt = 0.1; % interval between samples (secs)

% define integrator
% -------------------------------------------------------------------------
M.IS = @ode_LL;

% set priors
% -------------------------------------------------------------------------
M.pE = 0; % prior expectation (parameters)
M.pC = 1; % prior covariance  (parameters)
M.hE = 4; % prior expectation (hyperparameters)
M.hC = 1; % prior covariance (hyperparameters)

% simulate data
% -------------------------------------------------------------------------
gp  = 0.5; % generative parameters
gh  = 5;   % generative hyperparameters

% Generate data
% -------------------------------------------------------------------------

% Integrate model over time to generate data
[y,t] = ode_LL(gp,M,U);

% Generative noise precision matrix
P = exp(gh)*eye(ny);

% Add noise
rng(1)
e = spm_normrnd(zeros(ny,1),spm_inv(P),1);  % noise
y = y + e;

% Pack
Y = struct();
Y.y = y;

%figure;plot(t,y,'.-');title('Simulated data');

% Invert model
% -------------------------------------------------------------------------
[Ep,Cp,Eh,Ch,F] = variational_laplace(M,U,Y);
%% Invert GLM using SPM NLSI for comparison (requires SPM)
M2   = struct();
M2.pE = M.pE; 
M2.pC = M.pC;
M2.hE = M.hE;
M2.hC = M.hC;
M2.G = @ode_LL;
M2.f = f;
M2.g = g;
M2.x = 1;
M2.noprint = true;
[Ep2,Cp2,Eh2,F2,~,~,~,Ch2] = spm_nlsi_GN(M2,U,Y);
%% Plot parameters and hyperparameters
figure;
subplot(2,2,1);
plot_parameters(Ep,Cp,gp);
ylim([0 0.7]);
axis square;
xlabel('Parameter');
title(sprintf('Parameters. F=%2.2f',F));
set(gca,'FontSize',12);

subplot(2,2,2);
plot_parameters(Eh,Ch,gh);
ylim([0 7]);
axis square;
xlabel('Hyperparameter');
title('Hyperparameters');
set(gca,'FontSize',12);

subplot(2,2,3);
plot_parameters(Ep2,Cp2,gp);
ylim([0 0.7]);
axis square;
xlabel('Parameter');
title(sprintf('Parameters - SPM. F=%2.2f',F2));
set(gca,'FontSize',12);

subplot(2,2,4);
plot_parameters(Eh2,Ch2,gh);
ylim([0 7]);
axis square;
xlabel('Hyperparameter');
title('Hyperparameters - SPM');
set(gca,'FontSize',12);

%% Plot fit to data
figure;
subplot(2,2,1);
plot(Y.y,'--');
hold on
plot(ode_LL(Ep,M,U));
xlabel('x'); ylabel('y');
axis square
set(gca,'FontSize',12);