clear
close all
% Navigate to a folder containing the data3D.csv you want to filter

% SX: Use a Recurrent Neural Network (RNN) to filter and smooth noisy 3D motion tracking data from a CSV file (data3D.csv).
% 1. Load and Preprocess Data
% 2. Create Training Data (Fake “Pseudo-Trials”)
% 3. Build the RNN
%	•	RNN structure: Input layer (size = number of features) + Two hidden layers: 40 units each + Output layer: size = number of features
%	•	Activation functions: tanh (1st hidden layer); linear (output layer)
%	•	Training objective: Sum of squared errors (standard reconstruction loss).
%	•	Some regularization: Small L2 penalty (weightcost = 1e-4) to prevent overfitting.
% 4. Train the RNN using Hessian-Free Optimization
% 5. Evaluate the Trained RNN
% 6. Visualize Filtering Results
% 7. Apply RNN to the Full Original Dataset
% 8. Save the Filtered Output

%% Example code for non-linear filtering of continuous tracking data
% This basic idea is that we use an RNN to auto-encode frame by frame tracking data,
% while also trying to minimize deviations for data points that are not temporally smooth
% or have low confidence

% In this code I do not fit data below a certain confidence threshold. If you were to implement this
% in PyTorch, it would be better to multiply the reconstruction error for every data point by
% the corresponding confidence for that point – thereby making correctness directly proportional
% to confidence. Let me know if this makes sense!

%% Load and pre-process
tbl_data = readtable('data3D.csv', 'NumHeaderLines', 0, 'ReadVariableNames', true);
nan_ind = find(isnan(table2array(tbl_data(:,1))));
use_data = table2array(tbl_data);
use_data(nan_ind,:) = 0;
orig_data = use_data;
norm_m = mean(use_data,1);
use_data = use_data - repmat(norm_m, [size(use_data,1) 1]);
norm_sd = std(use_data,1);
use_data = use_data ./ repmat(norm_sd, [size(use_data,1), 1]);
use_data(nan_ind,:) = 0;

%% Create training data
confidenceThreshold = 0.1; % data points below this confidence level will not be included as targets
numTrials = 400; % number of pseudo-trials
tL = 200; % number of frames per trial
predShift = 1; % shift predictions this many frames into the past. This gives the network a bit of a time buffer
inp = cell(1,numTrials); targ = cell(1,numTrials);
for tr = 1:numTrials
    ind = randsample(size(use_data,1)-tL,1);
    tRange = ind : ind + tL;
    inp{tr} = use_data(tRange,:)';
    while sum(inp{tr}(:)) == 0
        ind = randsample(size(use_data,1)-tL,1);
        tRange = ind : ind + tL;
        inp{tr} = use_data(tRange,:)';
    end
    shifted = tRange-predShift;
    shifted(shifted < 1) = 1;
    targ{tr} = use_data(shifted,:)';
    for j = 1 : size(use_data,2)/4
        ind = find(orig_data(tRange,j*4) < confidenceThreshold);
        targ{tr}((j*4)-3:(j*4)-1,ind) = nan;
        targ{tr}(j*4,:) = nan;
    end
end

%%  Construct network
N = 40;
B = size(targ{1},1);
I = size(inp{1},1);
dt = 1;
tau = 2;

layer_sizes = [I N N B];
layer_types = {'linear', 'tanh', 'linear'};
g = [1 1.3 1];
h = [1 1 1];
obj_fun_type = 'sum-of-squares';

net = init_rnn(layer_sizes, layer_types, g, obj_fun_type, 'tau', tau, 'dt', dt, 'numconn', Inf, 'dolearnstateinit', 1, 'dolearnbiases', 1, 'mu', 1, ...
    'costmaskfacbylayer', [1 1 1], 'modmaskbylayer', [1 1 1], ...
    'doinitstateinitrandom', 0, 'doinitstatebiasesrandom', 0, 'h', h, ...
    'netnoisesigma', 0, 'maintainDale', false); 
net.h = h;
net.g = g;

net.frobeniusNormRecRecRegularizer = 0;
net.firingRateMean.weight = 0;
net.firingRateMean.desiredValue = 0;
net.firingRateMean.mask = ones(N,1);
wc = 1e-4; % 1e-4 Best. This currently applies L2 to all network weights
net.wc = wc;

doParallel = true; % You really want to do this parallel
close all
[theta, objfun_train, objfun_test, stats] = hfopt2(net, inp, targ, ...
    [], [], ...
    'weightcost', wc, ...
    'Sfrac', 0.05, ... % Fraction of data to use in mini-batch
    'doplot', true, ...
    'doplotallobjectives', true, ...
    'displaylevel', 1, ...
    'maintainDale', false, ...
    'initlambda', 1e-4, ...
    'objfunmin', -Inf, ...
    'errtol', [0 0], ...
    'doparallelnetwork', doParallel, 'doparallelobjfun', doParallel, 'doparallelgradient', doParallel, 'doparallelcgafun', doParallel, ...
    'maxhfiters', 40);
net.theta = theta;

%% Process training data through trained network
net.noiseSigma = 0;
r = []; z = [];
RSqu = zeros(1,length(inp));
wc = net.wc;
N = size(net.originalX0s,1);
eval_network_rnn2 = create_eval_network_rnn2(wc);
eval_network = create_eval_network2(eval_network_rnn2, wc);
package = eval_network(net, inp, targ, 1, 1:length(inp), ...
    [], [], 'doparallel', true, 'dowrappers', false);
data = package{1};
for cond = 1:length(inp)
    RNNdata = data{cond};
    r(:,:,cond) = RNNdata{1};
    z(:,:,cond) = RNNdata{3};
    ind = ~isnan(targ{cond});
    RSqu(cond) = (sum((RNNdata{3}(ind) - targ{cond}(ind)).^2) / sum(targ{cond}(ind).^2));
end
disp(['Error: ' num2str(mean(RSqu))])
[n_Wru_v, n_Wrr_n, m_Wzr_n, n_x0_c, n_bx_1, m_bz_1] = unpackRNN(net, net.theta);


% Example filtered data for 6 pseudo-trials
figure(5)
clf
count = 1;
for j = 1:6
    subplot(6,2,count)
    plot(targ{j}')
    set(gca, 'XLim', [1 size(targ{j},2)])
    box off
    subplot(6,2,count+1)
    plot(z(:,:,j)')
    set(gca, 'XLim', [1 size(targ{j},2)])
    box off
    count = count + 2;
end

% Example overlapped comparison of data and filtered data
figure(6)
clf
count = 1;
for j = [1:3 21:23 29:31 41:43 53:55]
    subplot(5,3,count)
    plot(targ{1}(j,:)')
    hold on
    plot(z(j,:,1)')
    box off
    count = count + 1;
end


%% Process original dataset
all_data = {use_data'};
net.noiseSigma = 0;
z = [];
wc = net.wc;
N = size(net.originalX0s,1);
eval_network_rnn2 = create_eval_network_rnn2(wc);
eval_network = create_eval_network2(eval_network_rnn2, wc);
package = eval_network(net, all_data, [], 1, 1:length(all_data), ...
    [], [], 'doparallel', true, 'dowrappers', false);
data = package{1};
RNNdata = data{1};
out_data = RNNdata{3}';


%% Save filtered data in new table
tbl_data = readtable('data3D.csv', 'NumHeaderLines', 0, 'ReadVariableNames', true);
renorm_data = (out_data .* repmat(norm_sd, [size(out_data,1) 1])) + repmat(norm_m, [size(out_data,1) 1]);
% correct for missing header (1 line)
% correct predshift (-1 for header)
useShift = predShift - 1;
renorm_data(1:end-useShift,:) = renorm_data(useShift+1:end,:);
renorm_data(end-useShift+1:end,:) = repmat(renorm_data(end-useShift,:), [useShift, 1]);
orig_data = table2array(tbl_data);
renorm_data(nan_ind,:) = nan;
tbl_data(:,:) = array2table(renorm_data);
tbl_data(end+1,:) = tbl_data(end,:);
writetable(tbl_data, 'data3D_aug.csv');
