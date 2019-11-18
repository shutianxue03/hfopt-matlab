function SeqTask_Test(simparams)

assert(simparams.memRange(2) == 0, 'Must use fixed memory period')
assert(simparams.batchSize == 1, 'Must use single trial mode')

loadDir = [simparams.baseDir '/RNNs/' simparams.name '/'];
% Load best trained network
listing = dir([loadDir 'hfopt_*.mat']);
thisError = [];
for i = 1:length(listing)
    name = listing(i).name;
    iEnd = length(name)-8;
    for j = iEnd:-1:1
        if strcmp(name(j), '_')
            iBegin = j+1;
            break
        end
    end
    thisError(i) = str2double(name(iBegin:iEnd));
end
[~, loadInd] = min(thisError);
thisFile = listing(loadInd).name;

load(thisFile)
disp(['Loaded: ' thisFile])



simparams.forwardPass = [];

[inp, targ, did_change_train_data, net, all_simdata] = ...
    simparams.taskfun(net, [], [], simparams, [], true, true);

%net.noiseSigma = 0.02;
wc = net.wc;
eval_network_rnn2 = create_eval_network_rnn2(wc);
eval_network = create_eval_network2(eval_network_rnn2, wc);
package = eval_network(net, inp, targ, 1, 1:length(inp), ...
    [], [], 'doparallel', true, 'dowrappers', false);
data = package{1};
clear z r x targs inpBig
for cond = 1:length(inp)
    RNNdata = data{cond};
    % find targets
    targs(cond,:) = [find(inp{cond}(1:5,simparams.preTime+5) == 1) find(inp{cond}(1:5,simparams.preTime+15) == 1) ...
        find(inp{cond}(1:5,simparams.preTime+25) == 1) find(inp{cond}(1:5,simparams.preTime+35) == 1) find(inp{cond}(1:5,simparams.preTime+45) == 1)];
    tRange = 1:size(RNNdata{3},2);
    z(:,:,cond) = RNNdata{3}(:,tRange);
    r(:,:,cond) = RNNdata{1}(:,tRange,:);
    x(:,:,cond) = RNNdata{5}(:,tRange,:);
    inpCut = inp{cond}(:,simparams.preTime+10*simparams.numTargets+10);
    inpBig(:,:,cond) = inp{cond}(:,tRange);
    simparams.forwardPass{cond}{3} = RNNdata{3};
end
[inp, targ, did_change_train_data, net, all_simdata] = ...
    taskfun(net, inp, targ, simparams, [], false, true);

[n_Wru_v, n_Wrr_n, m_Wzr_n, n_x0_c, n_bx_1, m_bz_1] = unpackRNN(net, net.theta);


cmap = [lines(5); 0 0 0];

close all
figure(1)
clf
for i = 1:4
    subplot(3,4,i)
    hold on
    for j = [6 1:5]
        plot(inp{i}(j,:)', 'LineWidth', 2, 'Color', cmap(j,:))
    end
    set(gca, 'YLim', [-0.05 1.05])
    subplot(3,4,i+4)
    hold on
    plot(simparams.forwardPass{i}{3}')
    plot([1 size(simparams.forwardPass{i}{3},2)], [0.5 0.5], 'Color', [0.7 0.7 0.7])
    set(gca, 'YLim', [-0.05 1])
    subplot(3,4,i+8)
    plot(targ{i}')
    set(gca, 'YLim', [-0.05 1])
    box off
end


% t-sne
figure(2)
clf
count = 1;
Y = cell(1,simparams.numTargets);
for type = 1:simparams.numTargets
    for epoch = 1:simparams.numTargets
        if type == 1
            t = simparams.preTime + epoch*10;
            data = squeeze(r(:,t,:))';
            Y{epoch} = tsne(data, 'NumDimension', 3);
            %[~, Y{epoch}, ~] = pca(data);
        end
        
        subplot(simparams.numTargets,simparams.numTargets,count)
        hold on
        for cond = 1:size(Y{epoch},1)
            plot3(Y{epoch}(cond,1), Y{epoch}(cond,2), Y{epoch}(cond,3), '.', 'Color', cmap(targs(cond,type),:))
        end
        view(3)
        
        count = count + 1;
    end
end



pcData = r(:,:)';
[coeff, score, latent] = pca(pcData);
v = cumsum(latent) / sum(latent);
pc = reshape(score', size(r));

figure(3)
clf
hold on
for cond = 1:size(pc,3)
    plot3(pc(1,:,cond), pc(2,:,cond), pc(3,:,cond), 'Color', cmap(targs(cond,1),:))
end

nonLin = net.layers(2).transFun;

% Define plotRanges upfront
plotRange{1} = 1 : size(x,2);
ti{1} = [];
go = length(plotRange{1}) - simparams.moveTime;
endCue = simparams.preTime + simparams.numTargets*10 + 1;
plotRange{2} = size(x,2) - simparams.moveTime + 6 : size(x,2);
ti{2} = [];

clear fp_struct fpd FPs runLinear runLinearOutput
% Mem / Move
for epoch = 1
    fval_tol = 1e-15;
    while true
        if epoch == 1
            tRange = simparams.preTime + simparams.numTargets*10 + 1 : size(x,2) - simparams.moveTime - 10;
        elseif epoch == 2
            tRange = size(x,2) - simparams.moveTime + 6 : size(x,2);
        end
        
        xmeans = x(:,tRange,:);
        constantInput = inpCut;
        
        epsilon = 0.001;
        niters = 4;
        max_eps = 0.01;
        
        [fp_struct, fpd] = find_many_fixed(net, niters, xmeans(:,:), epsilon, max_eps, fval_tol, ...
            'constinput', constantInput, 'optmaxiters', 10000, 'display', 'off', ...
            'dotopomap', false, 'dobail', true, 'tolfun', fval_tol, 'tolx', 1e-25);
        
        fnorm = cell2mat({fp_struct.FPNorm});
        ind = find(~isnan(fnorm));
        if ~isempty(ind)
            FPs{epoch} = fp_struct(ind);
            
            thisT = plotRange{epoch}(1);
            runLength = length(plotRange{epoch}) - 1;
            for cond = 1:size(x,3)
                xmeans = x(:,thisT,cond);
                constantInput = inpCut;
                
                thisFP = FPs{epoch}(1).FP;
                
                [n_x_tp1, m_z_tp1]  = run_rnn_linear_dynamics(net, thisFP, constantInput, xmeans, runLength);
                runLinear{epoch}(:,:,cond) = n_x_tp1;
                runLinearOutput{epoch}(:,:,cond) = m_z_tp1;
            end
            break
        else
            disp('INCREASING FVAL TOL')
            fval_tol = fval_tol * 10;
        end
    end
end

disp('Number of FPs per epoch')
for epoch = 1
    disp(['epoch ' num2str(epoch) ': ' num2str(length(FPs{epoch}))])
end

cueSpeed = cell2mat({FPs{1}.FPVal});
[~, takeInd] = min(cueSpeed);
figure(5)
clf
set(gcf, 'Position', [50 50 275 220], 'PaperPositionMode', 'Auto')
for i = 1
    if i == 1
        useInd = takeInd;
    else
        useInd = 1;
    end
    xList = -1.5:0.5:0;
    xName = {'67','100','200','Inf'};
    yList = [-1.8850 -1.2567 -0.6283 0 0.6283 1.2567 1.8850]; %[-1.2567 -0.9425 -0.6283 -0.3142 0 0.3142 0.6283 0.9425 1.2567];
    yName = {'-6','-4','-2','0','2','4','6'};
    hold on
    plot(real(FPs{i}(useInd).eigenValues), imag(FPs{i}(useInd).eigenValues), '+', 'MarkerSize', 6, 'LineWidth', 1.5)
    set(gca, 'XTick', xList, 'XTickLabel', xName, 'YTick', yList, 'YTickLabel', yName)
    box off
    set(gca, 'YLim', [yList(1) yList(end)], 'XLim', [-2 0.3])
    xlabel('Decay half-life (ms)')
    ylabel('Oscillation frequency (Hz)')
    plot([0 0], get(gca, 'YLim'), 'Color', [0.6 0.6 0.6])
    set(gca, 'FontSize', 14)
end


plotFPs{1} = FPs{1}(takeInd);
%plotFPs{2} = FPs{2}(1);
%disp(['fixed point speeds: ' num2str(FPs{1}(1).FPVal) ' ' num2str(FPs{2}(1).FPVal)])


figure(6)
set(gcf, 'Position', [1290 550 600 420], 'PaperPositionMode', 'Auto')
clf
subCount = 1;
xD = 3;

data = nonLin(x);
preLabel = 'PC ';
coeffData = cell(1,2);
for part = 1
    hold on
    thisPlotRange = plotRange{part};
    thisTi = ti{part};
    
    pcData = data(:,thisPlotRange,:);
    
    pcData = pcData(:,:)';
    m = mean(pcData,1);
    meanData{part} = m;
    pcData = pcData - repmat(m, [size(pcData,1) 1]);
    
    [coeff, pcData2, latent] = pca(pcData);
    thisCoeff = coeff(:,1:3);
    coeffData{part} = thisCoeff;
    score = pcData * thisCoeff;
    
    pc = reshape(score', [3 length(thisPlotRange) size(x,3)]);
    savePC{part} = score;
    
    scaler = 1;
    FList = plotFPs{part};
    F = [];
    E1 = cell(1,length(FList)); E2 = cell(1,length(FList)); EColor = cell(1,length(FList));
    for i = 1:length(FList)
        thisFP = nonLin(FList(i).FP);
        F(i,:) =  (thisFP - m')' * coeffData{part};
        count = 1;
        eigenInd = 1;
        j = 1;
        while eigenInd <= 2
            if real(FList(i).eigenValues(j)) > 0
                stable = false;
            else
                stable = true;
            end
            if abs(imag(FList(i).eigenValues(j))) > 0
                [Q,R] = qr([real(FList(i).eigenVectors(:,j)) imag(FList(i).eigenVectors(:,j))]);
                for e = 1:2
                    thisEigenMinus = thisFP - (Q(:,e) * abs(FList(i).eigenValues(j)));
                    thisEigenPlus = thisFP + (Q(:,e) * abs(FList(i).eigenValues(j)));
                    E1{i}(count,:) = ((thisEigenMinus)-m')' * thisCoeff;
                    E2{i}(count,:) = ((thisEigenPlus)-m')' * thisCoeff;
                    if stable
                        EColor{i}(count,:) = [0.5 0.5 0.5];
                    else
                        EColor{i}(count,:) = [1 0 0];
                    end
                    count = count + 1;
                end
                j = j + 2;
            else
                thisEigenMinus = thisFP - (real(FList(i).eigenVectors(:,j)) * abs(FList(i).eigenValues(j)));
                thisEigenPlus = thisFP + (real(FList(i).eigenVectors(:,j)) * abs(FList(i).eigenValues(j)));
                E1{i}(count,:) = ((thisEigenMinus)-m')' * thisCoeff;
                E2{i}(count,:) = ((thisEigenPlus)-m')' * thisCoeff;
                if stable
                    EColor{i}(count,:) = [0.5 0.5 0.5];
                else
                    EColor{i}(count,:) = [1 0 0];
                end
                count = count + 1;
                j = j + 1;
            end
            eigenInd = eigenInd + 1;
        end
    end
    
    for cond = 1:size(pc,3)
        if xD == 3
            plot3(pc(1,:,cond), pc(2,:,cond), pc(3,:,cond), 'Color', cmap(targs(cond,1),:))
            plot3(pc(1,endCue,cond), pc(2,endCue,cond), pc(3,endCue,cond), '.', 'Color', 'black', 'MarkerSize', 15)
            plot3(pc(1,go,cond), pc(2,go,cond), pc(3,go,cond), '.', 'Color', [0.7 0.7 0.7], 'MarkerSize', 15)
            if ~isempty(thisTi)
                plot3(pc(1,thisTi,cond), pc(2,thisTi,cond), pc(3,thisTi,cond), '.', 'Color', [0.6 0.6 0.6], 'MarkerSize', 15)
            end
        elseif xD == 2
            plot(pc(1,:,cond), pc(2,:,cond), 'Color', cmap(targs(cond,1),:))
            plot(pc(1,1,cond), pc(2,1,cond), '.', 'Color', 'black', 'MarkerSize', 10)
            if ~isempty(thisTi)
                plot(pc(1,thisTi,cond), pc(2,thisTi,cond), '.', 'Color', [0.6 0.6 0.6], 'MarkerSize', 10)
            end
        end
        
    end
    
    pointMap = repmat([0 0 0], [size(F,1) 1]);
    for i = 1:size(F,1)
        if xD == 3
            plot3(F(i,1),F(i,2),F(i,3),'+', 'MarkerSize', 10, 'MarkerFaceColor', 'none', 'MarkerEdgeColor', pointMap(i,:), 'LineWidth', 2)
            for j = 1:size(E1{i},1)
                plot3([E1{i}(j,1), E2{i}(j,1)], [E1{i}(j,2), E2{i}(j,2)], ...
                    [E1{i}(j,3), E2{i}(j,3)], 'LineWidth', 2, 'Color', EColor{i}(j,:))
            end
        else
            plot(F(i,1),F(i,2),'+', 'MarkerSize', 10, 'MarkerFaceColor', 'none', 'MarkerEdgeColor', pointMap(i,:), 'LineWidth', 2)
            for j = 1:size(E1{i},1)
                plot([E1{i}(j,1), E2{i}(j,1)], [E1{i}(j,2), E2{i}(j,2)], ...
                    'LineWidth', 2, 'Color', EColor{i}(j,:))
            end
        end
        
    end
    xlabel([preLabel '1'])
    ylabel([preLabel '2'])
    if xD == 3
        zlabel([preLabel '3'])
        view(3)
        axis vis3d
    end
    subCount = subCount + 1;
end


figure(7)
set(gcf, 'Position', [1407 17 490 430], 'PaperPositionMode', 'Auto')
clf
subCount = 1;
xD = 3;
for part = 1
    hold on
    
    pcData = nonLin(runLinear{part});
    pcData = pcData(:,:)';
    m = meanData{part};
    pcData = pcData - repmat(m, [size(pcData,1) 1]);
    thisData = pcData;
    thisCoeff = coeffData{part};
    score = thisData * thisCoeff;
    
    pc = reshape(score', [3 size(runLinear{part},2) size(x,3)]);
    
    scaler = 1;
    FList = plotFPs{part};
    F = [];
    E1 = cell(1,length(FList)); E2 = cell(1,length(FList));
    for i = 1:length(FList)
        thisFP = nonLin(FList(i).FP);
        F(i,:) = ((thisFP)-m')' * thisCoeff;
        count = 1;
        eigenInd = 1;
        j = 1;
        while eigenInd <= 2
            if real(FList(i).eigenValues(j)) > 0
                stable = false;
            else
                stable = true;
            end
            if abs(imag(FList(i).eigenValues(j))) > 0
                [Q,R] = qr([real(FList(i).eigenVectors(:,j)) imag(FList(i).eigenVectors(:,j))]);
                for e = 1:2
                    thisEigenMinus = thisFP - (Q(:,e) * abs(FList(i).eigenValues(j)));
                    thisEigenPlus = thisFP + (Q(:,e) * abs(FList(i).eigenValues(j)));
                    E1{i}(count,:) = ((thisEigenMinus)-m')' * thisCoeff;
                    E2{i}(count,:) = ((thisEigenPlus)-m')' * thisCoeff;
                    if stable
                        EColor{i}(count,:) = [0.5 0.5 0.5];
                    else
                        EColor{i}(count,:) = [1 0 0];
                    end
                    count = count + 1;
                end
                j = j + 2;
            else
                thisEigenMinus = thisFP - (real(FList(i).eigenVectors(:,j)) * abs(FList(i).eigenValues(j)));
                thisEigenPlus = thisFP + (real(FList(i).eigenVectors(:,j)) * abs(FList(i).eigenValues(j)));
                E1{i}(count,:) = ((thisEigenMinus)-m')' * thisCoeff;
                E2{i}(count,:) = ((thisEigenPlus)-m')' * thisCoeff;
                if stable
                    EColor{i}(count,:) = [0.5 0.5 0.5];
                else
                    EColor{i}(count,:) = [1 0 0];
                end
                count = count + 1;
                j = j + 1;
            end
            eigenInd = eigenInd + 1;
        end
    end
    
    
    hold on
    for cond = 1:size(pc,3)
        if xD == 3
            plot3(pc(1,:,cond), pc(2,:,cond), pc(3,:,cond), 'Color', cmap(targs(cond,1),:))
            plot3(pc(1,1,cond), pc(2,1,cond), pc(3,1,cond), '.', 'Color', 'black', 'MarkerSize', 10)
        elseif xD == 2
            plot(pc(1,:,cond), pc(2,:,cond), 'Color', cmap(targs(cond,1),:))
            plot(pc(1,1,cond), pc(2,1,cond), '.', 'Color', 'black', 'MarkerSize', 10)
        end
    end
    pointMap = repmat([0 0 0], [size(F,1) 1]);
    for i = 1:size(F,1)
        if xD == 3
            plot3(F(i,1),F(i,2),F(i,3),'+', 'MarkerSize', 10, 'MarkerFaceColor', 'none', 'MarkerEdgeColor', pointMap(i,:), 'LineWidth', 2)
            for j = 1:size(E1{i},1)
                plot3([E1{i}(j,1), E2{i}(j,1)], [E1{i}(j,2), E2{i}(j,2)], ...
                    [E1{i}(j,3), E2{i}(j,3)], 'LineWidth', 2, 'Color', EColor{i}(j,:))
            end
        else
            plot(F(i,1),F(i,2),'+', 'MarkerSize', 10, 'MarkerFaceColor', 'none', 'MarkerEdgeColor', pointMap(i,:), 'LineWidth', 2)
            for j = 1:size(E1{i},1)
                plot([E1{i}(j,1), E2{i}(j,1)], [E1{i}(j,2), E2{i}(j,2)], ...
                    'LineWidth', 2, 'Color', EColor{i}(j,:))
            end
        end
    end
    xlabel('PC 1')
    ylabel('PC 2')
    if xD == 3
        zlabel('PC 3')
        view(3)
        axis vis3d
    end
    subCount = subCount + 1;
end
end