%% VL Scheme - General linear model example

% define model
% -------------------------------------------------------------------------
M = struct();
g = @(P,M,U) M.X*P;
M.g = g;

% set priors
% -------------------------------------------------------------------------
M.pE = [0; 0];       % prior expectation
M.pC = eye(2) .* 1;  % prior covariance  (parameters)
M.hE = [4 4]';       % prior expectation (hyperparameters)
M.hC = eye(2) .* 1;  % prior covariance  (hyperparameters)

% define precision components
% -------------------------------------------------------------------------
ny  = 100;                              % number of data points
Y   = struct();
Y.Q = {diag([ones(ny/2,1); zeros(ny/2,1)]);
       diag([zeros(ny/2,1); ones(ny/2,1)])};

% simulate data
% -------------------------------------------------------------------------
gp  = [0.5; 0.1];                       % generative parameters
gh  = [2; 6];                           % generative hyperparameters

% Generate data
% -------------------------------------------------------------------------

% Generative precision matrix
P = zeros(ny,ny);
for i = 1:length(Y.Q)
    P = P + exp(gh(i))*Y.Q{i};
end

% Design matrix for GLM (constant and gradient)
M.X = [ones(ny,1) linspace(1,100,ny)'];
M.X(:,2) = M.X(:,2) - mean(M.X(:,2));

% Simulate data and add noise
rng(1)
e   = spm_normrnd(zeros(ny,1),spm_inv(P),1);  % noise
Y.y   = g(gp,M) + e;
U   = [];

% Invert
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
%% Plot
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

%% Plot parameters and hyperparameters
figure;
subplot(2,2,1);
plot(Y.y,'--');
hold on
plot(M.g(Ep,M,U));
xlabel('x'); ylabel('y');
axis square
set(gca,'FontSize',12);
%% Bayesian model comparison

% Invert model with one precision component
M.hE = 4; 
M.hC = 1;
Y.Q  = {eye(ny)};
[~,~,~,~,F_1Q] = variational_laplace(M,U,Y);

% Invert model with two precision components
F_2Q = F;

% Invert model with three precision components
M.hE = [4;4;4]; 
M.hC = eye(3);
Y.Q            = {blkdiag(eye(33),zeros(67));
                  blkdiag(zeros(33),eye(33),zeros(34));
                  blkdiag(zeros(66),eye(34))};
[~,~,~,~,F_3Q] = variational_laplace(M,U,Y);

% Compute log Bayes factor for each model relative to the worst model
logBF = [F_1Q F_2Q F_3Q];
logBF = logBF - min(logBF);

% Compute posterior probability for each model (softmax)
P = exp(logBF')/sum(exp(logBF'));

figure;
subplot(2,2,1);
bar(logBF); 
xlabel('Model m');
set(gca,'FontSize',12);
axis square;
title('Relative F');

subplot(2,2,2);
bar(P);
xlabel('Model m');ylabel('P(m|y)');
axis square;
set(gca,'FontSize',12);
title('Model posterior');