function SeqTask_Train(simparams, doPlot)

baseDir = simparams.baseDir;
N = simparams.N;
B = simparams.B;
I = simparams.I;
g = simparams.g;
dt = simparams.dt;
tau = simparams.tau;

layer_sizes = [I N N B];
h = [1 1 1];
net = init_rnn(layer_sizes, simparams.layer_types, g, simparams.obj_fun_type, 'tau', tau, 'dt', dt, 'numconn', Inf, 'dolearnstateinit', 1, 'dolearnbiases', 1, 'mu', 1, ...
    'costmaskfacbylayer', [1 0 1], 'modmaskbylayer', [1 1 1], ...
    'doinitstateinitrandom', 0, 'doinitstatebiasesrandom', 0, 'h', h, ...
    'netnoisesigma', 0);
net.hasCanonicalLink = true;

net.h = h;
net.g = g;

net.frobeniusNormRecRecRegularizer = simparams.Frob;
net.firingRateMean.weight = simparams.firingrate;
net.firingRateMean.desiredValue = 0;
net.firingRateMean.mask = ones(N,1);
wc = simparams.wc;
net.wc = wc;

% Random initialization of weights. 
[n_Wru_v, n_Wrr_n, m_Wzr_n, n_x0_c, n_bx_1, m_bz_1] = unpackRNN(net, net.theta);

m_bz_1 = randn(size(m_bz_1)) / 1e4;
n_x0_c = randn(size(n_x0_c)) / 1e4;
n_bx_1 = randn(size(n_bx_1)) / 1e4;
m_Wzr_n = randn(size(m_Wzr_n)) / sqrt(size(m_Wzr_n,2));

e = abs(eig(n_Wrr_n));
maxE = max(e);
%n_Wrr_n = n_Wrr_n / maxE * g;
%e = abs(eig(n_Wrr_n));
%maxENew = max(e);

net.theta = packRNN(net, n_Wru_v, n_Wrr_n, m_Wzr_n, n_x0_c, n_bx_1, m_bz_1);
%net.modMask(net.theta == 0) = 0;

doParallel = true;

close all
mkdir([baseDir '/RNNs/' simparams.name])
simparams.forwardPass = [];
[~, ~, ~, ~] = hfopt2(net, simparams.taskfun, [], ...
    [], [], ...
    'weightcost', net.wc, ... % Cost on weights (wIn^2 + wOut^2)
    'Sfrac', 1, ... % Fraction of data to use in mini-batch
    'savepath', [baseDir '/RNNs/' simparams.name '/'], ... % Where to save iterations %
    'saveevery', simparams.saveEvery, ...
    'filenamepart', simparams.name, ... % Unique savename identifier
    'optplotfun', simparams.plotfun, ...
    'doplotallobjectives', true, ...
    'displaylevel', 1, ...
    'maintainDale', false, ...
    'initlambda', 1e-4, ...
    'objfuntol', 1e-6, ...
    'errtol', [0.0 0.0], ...
    'miniter', 30, ...
    'doparallelnetwork', doParallel, 'doparallelobjfun', doParallel, 'doparallelgradient', doParallel, 'doparallelcgafun', doParallel, ...
    ...   'optevalfun', @GNM_optional_eval_fun, ...
    'maxhfiters', 1000, ...
    'doplot', doPlot, ...
    'simparams', simparams);
end