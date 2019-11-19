%%% Main function for simulating basic sequence tasks %%%
%%% November 18, 2019 %%%
%%% Jonathan Michaels %%%

% Define set of simulation parameters
simparams = [];
% Top directory for project
simparams.baseDir = '/Users/jonathanamichaels/Desktop/jmichaels/Projects/ST';
% Specific title of this run. Simulations will be saved their.
simparams.name = 'Test';
% Which task to run
simparams.taskfun = @SeqTask_taskfun_gaussian; % or @SeqTask_taskfun_threshold
% Plotting function during training
simparams.plotfun = @SeqTask_hfopt_Plot;

simparams.numTrials = 25; % Trials to simulate
simparams.numTargets = 5; % Goal targets to simulate. Please stick to 5 for now.
simparams.batchSize = 3; % How many trials to string together
simparams.memRange = [10 90]; % Variability of memory period. Mem = memRange(1) + randuniform(memRange(2))
simparams.preTime = 10; % Time before visual cues
simparams.moveTime = 100; % Total movement time
simparams.noGoFreq = 0.1; % Frequency of nogo trials

simparams.N = 300; % Number of neurons
simparams.B = simparams.numTargets; % Number of outputs
simparams.I = simparams.numTargets + 1; % Number of inputs

% dt / tau determines whether or not the network is discrete (dt = tau) or continuous time (dt / tau < 1)
simparams.dt = 1;
simparams.tau = 10;

simparams.layer_types = {'linear', 'recttanh', 'rectlinear'}; % input / hidden / output layer activation functions
% if g is too large then the network can't handle very variable memory periods
simparams.g = [1 1.1 1]; % spectral scaling of each layer
simparams.obj_fun_type = 'sum-of-squares'; % type of error

simparams.wc = 0; % cost on square of input and output weights
simparams.firingrate = 0; % cost on firing rate
simparams.Frob = 0; % cost on trajectory complexity

doPlot = true; % Do you want to watch training as it goes?

% LET'S TRAIN
SeqTask_Train(simparams, doPlot)

% LET'S VISUALIZE RESULTS
% We'll look for trained networks based on how you named this batch (simparams.name)
simparams.numTrials = 100;
simparams.batchSize = 2; % Must use double trial mode for plotting purposes
simparams.memRange = [80 0]; % Must use a fixed memory period for plotting purposes
simparams.noGoFreq = 0.1;

SeqTask_Test(simparams)

