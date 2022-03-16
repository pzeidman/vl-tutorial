% Runs all demos of the Variational Laplace scheme. Note that the demos
% compare the output of this standalone toolbox against the output of
% the canonical implementation of the VL scheme in the SPM software package.
% This requires having SPM on your Matlab path. It can be downloaded from 
% https://www.fil.ion.ucl.ac.uk/spm/software/download/ 

% Get current path
example_dir = fileparts(mfilename('fullpath'));
if isempty(example_dir)
    example_dir = fileparts(pwd);
end

% Add VL scripts to path
toolbox_dir = fullfile(example_dir,'..','toolbox');
addpath(toolbox_dir);

% Run all demos
run_VL_GLM1
run_VL_GLM2
run_VL_exp
run_VL_haemodynamic

