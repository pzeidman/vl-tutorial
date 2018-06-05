function vl_demo
% Demo of evaluating the free energy of a model fitted to timeseries.
%
% The model has the form:
%
%     dz/dt = ( -0.5*exp(a) )z + cu
%     y     = z + e
%
%     e     ~ N(0, 1/exp(lambda))
%
% There are three parameters. Parameter a is a log scaling parameter which 
% scales up or down the default rate of decay -0.5Hz. Parameter c is the 
% sensitivity of the system to input. Hyperparameter lambda is the log 
% precision of the observatiom noise.
%
% Key variables:
%
% M.pE.f - Prior expectation of parameters
% M.pE.h - Prior expectation of hyperparameters
% M.pC.f - Prior (co)variance of parameters
% M.pC.h - Prior (co)variance of hyperparameters
%
% Ep.f   - Posterior expectation of parameters
% Ep.h   - Posterior expectation of hyperparameters
% Cp.f   - Posterior (co)variance of parameters
% Cp.h   - Posterior (co)variance of parameters
%
% U      - input timing (delta functions)
%
% Peter Zeidman, Ashwini Oswal, Sukhbinder Kumar
% _________________________________________________________________________

% Seed random number generator
rng(1);

% Number of time points (seconds). For simplicity we assume sampling is at
% the same rate as the integration steps.
num_steps = 300;

% Set priors and model settings
M = struct();
M.pE.f.A      = 0; M.pC.f.A = 1/64;
M.pE.f.C      = 0; M.pC.f.C = 1;
M.pE.h.lambda = 6; M.pC.h.lambda = 1;
M.num_steps   = num_steps;

% Manually set posteriors (for simulating data)
Ep.f.A      = -0.5; 
Ep.f.C      = 1;
Ep.h.lambda = 2;
Cp.f.A      = M.pC.f.A ./ 16; 
Cp.f.C      = M.pC.f.C ./ 16; 
Cp.h.lambda = M.pC.h.lambda   ./ 16; 

% Set input timeseries (a delta function every 20s)
U = zeros(num_steps,1);
U(1:20:600) = 1;

% Generate simulated data with noise
y  = integrate_model(Ep,M,U);
sd = 1 / exp(Ep.h.lambda);
e  = randn(1,num_steps) * sd;
Y  = y + e;

% Plot data
figure;
subplot(2,3,1:3);
plot(Y);
title('Simulated data: $$dz/dt= az + cu $$','interpreter','latex','FontSize',16);
xlabel('Time (secs)'); ylabel('Measurement');
drawnow;

% Select range of parameters
a = linspace(-1,-0.1,50);
c = linspace(0,2,50);
[A,C] = meshgrid(a,c);

% Compute free energy landscape
disp('Performing grid search...');
F = zeros(size(A));
accuracy   = zeros(size(A));
complexity = zeros(size(A));
for row = 1:size(A,1)
    for col = 1:size(A,2)
        Ep.f.A = A(row,col);
        Ep.f.C = C(row,col);
        [F(row,col),accuracy(row,col),complexity(row,col)] = free_energy(Y,Ep,Cp,M,U);
    end
end

% Identify optimal parameters from grid search
[~,idx] = max(accuracy(:));
[maxrow,maxcol] = ind2sub(size(F),idx);

% Plot Free Energy landscape
subplot(2,3,4);
imagesc(a,c,F);hold on; contour(a,c,F,'Color','k');
ylabel('C'); xlabel('A');
text(a(maxcol),c(maxrow),'+','Color','r');
fprintf('Optimal parameters: a=%2.2f c=%2.2f\n',a(maxcol),c(maxrow));
axis square;
title('Free energy');

% Plot accuracy landscape
subplot(2,3,5);
imagesc(a,c,accuracy);hold on; contour(a,c,accuracy,'Color','k');
text(a(maxcol),c(maxrow),'+','Color','r');
ylabel('C'); xlabel('A');
axis square;
title('Accuracy');

% Plot complexity landscape
subplot(2,3,6);
imagesc(a,c,complexity);hold on; contour(a,c,complexity,'Color','k');
text(a(maxcol),c(maxrow),'+','Color','r');
ylabel('C'); xlabel('A');
axis square;
title('Complexity');
drawnow;

% State-space (e.g. neural) function: dz/dt = (-0.5 * exp(a))z + cu
% -------------------------------------------------------------------------
function dzdt = fz(z,t,P,M,U)
A = P.f.A;
C = P.f.C;
dzdt = -0.5*exp(A)*z + C*U(t,:);

% Observation function: y = z
% -------------------------------------------------------------------------
function y = gz(z,P,M,U)
y = z;

% Integration routine (Euler's method)
% -------------------------------------------------------------------------
function [y,z] = integrate_model(P,M,U)

z = zeros(1,M.num_steps);
y = zeros(1,M.num_steps);

h = 1;
for n = 1:(M.num_steps)
    z(n+1) = z(n) + h * fz(z(n),n,P,M,U);
    y(n+1) = gz(z(n+1),P,M,U);
end
z = z(2:end);
y = y(2:end);

% Calculate negative variational free energy
% -------------------------------------------------------------------------
function [F,accuracy,complexity] = free_energy(Y,Ep,Cp,M,U)

if isvector(Ep), Ep = spm_unvec(Ep,M.pE); end
if isvector(Cp), Cp = spm_unvec(Cp,M.pC); end

% Integrate model to get predicted timeseries
y = integrate_model(Ep,M,U);

% Unpack priors (parameters)
pEf = spm_vec(M.pE.f);
pCf = diag(spm_vec(M.pC.f));

% Unpack priors (hyperparameters)
pEh = spm_vec(M.pE.h);
pCh = diag(spm_vec(M.pC.h));

% Unpack posteriors (parameters)
Epf = spm_vec(Ep.f);
Cpf = diag(spm_vec(Cp.f));

% Unpack posteriors (hyperparameters)
Eph = spm_vec(Ep.h);
Cph = diag(spm_vec(Cp.h));

% Number of data points
N = length(Y);
    
% Compute prior precision terms
pY = exp(pEh)*eye(N); % data
pP = spm_inv(pCf);      % parameters
pH = spm_inv(pCh);    % hyperparameters

% Compute error terms
ey = (Y-y)';       % data
ep = Epf - pEf;      % parameters
eh = Eph - pEh;  % hyperparameters

% Free energy
accuracy   = -(0.5 * ey' * pY * ey) - (0.5 * logdet(inv(pY)))  - ((N/2) * log(2 * pi));
complexity = (0.5 * ep' * pP * ep)  + (0.5 * logdet(pCf))      - (0.5 * logdet(Cpf)) ...
            +(0.5 * eh' * pH * eh)  + (0.5 * logdet(pCh))      - (0.5 * logdet(Cph));

F = accuracy - complexity;  

% Log determinate of a matrix
% -------------------------------------------------------------------------
function ld = logdet(X)
ld = spm_logdet(X);