# A primer on Variational Laplace
This code accompanies the paper **A Primer on Variational Laplace** by Peter Zeidman, Karl Friston and Thomas Parr. <https://doi.org/10.31219/osf.io/28vwh>

## Standalone VL implementation and examples
A standalone implementation of the algorithm for MATLAB or Octave can be found in /matlab/toolbox/variational_laplace.m . Note that the example code in /matlab/examples/ compares the output of the standalone implementation against that of the [SPM toolbox](https://www.fil.ion.ucl.ac.uk/spm/software/download/), so SPM needs to be on the MATLAB path for the example code to run to completion.

Instructions:

1. Download the matlab/ folder to somewhere convenient on your computer
2. Using MATLAB or Octave, change the working directory to the matlab/examples/ folder
3. Run the script run_VL_all.m to reproduce the figures in the paper

## Attention to Visual Motion example
The Attention to Visual Motion example requires Dynamic Causal Modelling (DCM) for fMRI, implemented in the SPM software package. To run this example:

1. Download and install the [SPM toolbox](https://www.fil.ion.ucl.ac.uk/spm/software/download/).
2. Download and run the MATLAB script from the bottom of the [SPM data set webpage](https://www.fil.ion.ucl.ac.uk/spm/data/attention/). This will download the data and run the analysis.
