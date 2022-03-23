%% VL Scheme - General linear model example

% define model
% -------------------------------------------------------------------------
M = struct();
g = @(P,M,U) M.X*P;
M.g = g;

% set priors
% -------------------------------------------------------------------------
M.pE = [0; 0];      % prior expectation (parameters)
M.pC = eye(2) .* 1; % prior covariance  (parameters)
M.hE = 4;           % prior expectation (hyperparameters)
M.hC = 1;           % prior covariance (hyperparameters)

% define precision components
% -------------------------------------------------------------------------
ny  = 100;                              % number of data points
Y   = struct();
Y.Q = {eye(ny)};

% Simulate data
% -------------------------------------------------------------------------
gp  = [0.5; 0.1];                      % generative parameters
gh  = 3;                               % generative hyperparameters

% Generative precision matrix
P = zeros(ny,ny);
for i = 1:length(Y.Q)
    P = P + exp(gh(i))*Y.Q{i};
end

% Design matrix for GLM (constant and gradient)
M.X = [ones(ny,1) linspace(1,100,ny)']; % domain
M.X(:,2) = M.X(:,2) - mean(M.X(:,2));

% Simulate data and add noise
rng(1)
e   = spm_normrnd(zeros(ny,1),spm_inv(P),1);
Y.y   = g(gp,M) + e;
U   = [];

% Invert model
% -------------------------------------------------------------------------
[Ep,Cp,Eh,Ch,F] = variational_laplace(M,U,Y);

%% Invert GLM using SPM NLSI for comparison (requires SPM)
M2   = struct();
M2.pE = M.pE; 
M2.pC = M.pC;
M2.hE = M.hE;
M2.hC = M.hC;
M2.X  = M.X;
M2.G = g;
M2.noprint = true;
[Ep2,Cp2,Eh2,F2,~,~,~,Ch2] = spm_nlsi_GN(M2,U,Y);
%% Plot parameters and hyperparameters
figure;
subplot(2,2,1);
plot_parameters(Ep,Cp,gp);
ylim([0 0.6]);
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
ylim([0 0.6]);
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
plot(M.g(Ep,M,U));
xlabel('x'); ylabel('y');
axis square
set(gca,'FontSize',12);