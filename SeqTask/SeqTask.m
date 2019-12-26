function varargout=SeqTask(what,varargin)
% Different functions for the evaluation of
% Different neuronal networks
baseDir = '/Users/jdiedrichsen/Dropbox (Diedrichsenlab)/projects/RNN_sequence';

switch(what)
    case 'Run_all'
        % Train a network
        simparams=SeqTask('Get_simparamsTrain'); % Get parameters
        % Make modifications as desired
        SeqTask('trainNetwork',simparams); % Train a network
        
        % Now test it on single trials
        simparams=SeqTask('Get_simparamsTest');
        [D,v_inputtrain_T,m_targettrain_T]=SeqTask('Get_singleTrialTest',simparams);
        net = SeqTask('Get_network','GaussianNoRep2');
        data = SeqTask('Run_simulation',net,v_inputtrain_T); % Run the simulation of the current network
        D=SeqTask('performance',data,D);
    case 'Get_simparamsTrain'
        simparams.name = 'GaussianNoRep2';
        % How often to save a checkpoint of training
        simparams.saveEvery = 5;
        % Which task to run
        simparams.taskfun = @SeqTask_taskfun_gaussian; % or @SeqTask_taskfun_threshold
        % Plotting function during training
        simparams.plotfun = @SeqTask_hfopt_Plot;
        
        simparams.numEpisodes = 20;      % Number of Episodes to simulation
        simparams.numTrials = 3;        % How many trials to string together
        simparams.numTargets = 5;       % Number of elements in the sequence
        simparams.withreplace = false;   % Targets can repeat?
        simparams.memPeriod = [10 100];  % Range of memory period.
        simparams.preTime = 10; % Time before visual cues
        simparams.cueOn   = 8; % How long is each sequence cue on?
        simparams.cueOff   = 2; % How long is each sequence cue off?
        simparams.forceWidth= 25; % How long is each force press?
        simparams.forceIPI= 10;   % How long between onsets of each press?
        simparams.RT = 12;         % From onset of go-cue to beginning of force production
        simparams.moveTime = 100; % (simparams.numTargets-1)*simparams.forceIPI+simparams.forceWidth+30; % Total movement time
        simparams.noGoFreq = 0.1; % Frequency of nogo trials
        
        % Network size parameters
        simparams.N = 100; % Number of neurons
        simparams.B = 5;   % Number of outputs (5 fingers)
        simparams.I = simparams.B + 1; % desired output + go signal
        simparams.layer_types = {'linear', 'recttanh', 'rectlinear'}; % input / hidden / output layer activation functions
        % if g is too large then the network can't handle very variable memory periods
        simparams.g = [1 1.3 1]; % spectral scaling of each layer
        simparams.obj_fun_type = 'sum-of-squares'; % type of error
        
        % dt / tau determines whether or not the network is discrete (dt = tau) or continuous time (dt / tau < 1)
        simparams.dt = 1;
        simparams.tau = 10;
        
        % Network regularization
        simparams.wc = 0; % cost on square of input and output weights
        simparams.firingrate = 0; % cost on firing rate
        simparams.Frob = 0; % cost on trajectory complexity
        varargout={simparams};
    case 'trainNetwork'
        doPlot = true; % Make a plot every timestep
        doParallel = true; % Use parallel pool for training
        simparams=varargin{1};
        N = simparams.N;
        B = simparams.B;
        I = simparams.I;
        g = simparams.g;
        dt = simparams.dt;
        tau = simparams.tau;
        
        % Make the network from scratch
        layer_sizes = [I N N B];
        h = [1 1 1];
        net = init_rnn(layer_sizes, simparams.layer_types, g, simparams.obj_fun_type, ...
            'tau', tau, 'dt', dt, 'numconn', Inf, 'dolearnstateinit', 1, 'dolearnbiases', 1, 'mu', 1, ...
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
        
        % Random initialization of weights: Unpack and repack parameters of
        % RNN
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
            'doparallelnetwork', doParallel, 'doparallelobjfun', doParallel, ...
            'doparallelgradient', doParallel, 'doparallelcgafun', doParallel, ...
            ...   % 'optevalfun', @GNM_optional_eval_fun, ...
            'maxhfiters', 1000, ...
            'doplot', doPlot, ...
            'simparams', simparams);
    case 'Get_simparamsTest'
        % Generates simulation parameters for single trial tyes
        simparams.numTargets=5;
        simparams.numTrials=1;
        simparams.memPeriod=80; % Fixed memory period
        simparams.numEpisodes = 20;      % Number of Episodes to simulation
        simparams.numTrials = 3;        % How many trials to string together
        simparams.numTargets = 5;       % Number of elements in the sequence
        simparams.withreplace = false;   % Targets can repeat?
        simparams.memPeriod = 80;  % Range of memory period.
        simparams.preTime = 10; % Time before visual cues
        simparams.cueOn   = 8; % How long is each sequence cue on?
        simparams.cueOff   = 2; % How long is each sequence cue off?
        simparams.forceWidth= 25; % How long is each force press?
        simparams.forceIPI= 12;   % How long between onsets of each press?
        simparams.RT = 12;         % From onset of go-cue to beginning of force production
        simparams.moveTime = (simparams.forceWidth+simparams.forceIPI)*simparams.numTargets+20; % Total movement time
        simparams.noGoFreq=0;
        varargout={simparams};
    case 'Get_singleTrialTest'
        % Get an exhaustive set of trials
        simparams=varargin{1};
        D.targs=perms([1:5]);
        N=size(D.targs,1);
        D.episode = [1:N]';
        D.trial = ones(N,1);
        D.noGo = zeros(N,1);
        D.memLength = ones(N,1)*simparams.memPeriod(1);
        [v_inputtrain_T,D]=SeqTask_taskfun_inputs(D,simparams);
        m_targettrain_T=SeqTask_taskfun_outputs(D,simparams);
        varargout={D,v_inputtrain_T,m_targettrain_T};
    case 'Get_network'
        name = varargin{1};
        loadDir = [baseDir '/RNNs/' name '/'];
        % If no number is given, just get the network with the lowest error
        if (length(varargin)==1)
            listing = dir(fullfile(loadDir,'hfopt_*.mat'));
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
        else % Otherwise get one specific one
            networknum = varargin{2};
            listing = dir(fullfile(loadDir,sprintf('hfopt_%s_%d_*.mat',name,networknum)));
            if (length(listing)==0)
                error('file not found');
            else
                thisFile = listing(1).name;
            end;
        end;
        load(fullfile(loadDir,thisFile),'net','simparams')
        disp(['Loaded: ' thisFile])
        varargout={net,simparams};
    case 'Plot_Learningcurve'
        networks = {'GaussianNoRep1','GaussianNoRep2','GaussianTest'};
        E=[];
        for j=1:length(networks)
            loadDir = [baseDir '/RNNs/' networks{j} '/'];
            % Get all learning stages
            listing = dir(fullfile(loadDir,'hfopt_*.mat'));
            D=[];
            for i = 1:length(listing)
                name = listing(i).name;
                a=textscan(name,['hfopt_' networks{j} '_%d_%f']);
                D.network{i,1}=networks{j};
                D.batch(i,1)=double(a{1});
                D.error(i,1)=a{2};
                % T=load(name);
                % E=T.net.taskData;
            end
            E=addstruct(E,D);
        end;
        [a,b,c]=unique(E.network);
        lineplot(E.batch,E.error,'split',E.network,'leg','auto','style_thickline');
    case 'Run_simulation'
        % Run the network on the given input
        % Output:
        % Data{episode}{1} = Firing of latent units across time (after nonlinearity)
        % Data{episode}{3} = Firing rate of output units
        % Data{episode}{5} = Activity of latent units (before nonlinearity)
        
        net=varargin{1};
        inp=varargin{2};
        
        wc = net.wc;
        eval_network_rnn2 = create_eval_network_rnn2(wc);
        eval_network = create_eval_network2(eval_network_rnn2, wc);
        package = eval_network(net, inp, [], 1, 1:length(inp), ...
            [], [], 'doparallel', true, 'dowrappers', false);
        data = package{1};
        varargout = {data};
    case 'performance'
        % Rate performance and establish timing
        % D=SeqTask('performance',data,D);
        data = varargin{1};
        D= varargin{2};
        thresh =0.1;
        for i=1:length(data)
            out=data{i}{3};
            pressed=zeros(5,1);
            press = 0;
            for t=D.gocue(i):size(out,2);
                fing = find(pressed==0 & out(:,t)>thresh);
                if ~isempty(fing)
                    press = press+1;
                    D.press(i,press)=fing(1);
                    D.pressOnset(i,press)=t;
                    [~,D.pressMax(i,press)]=max(out(fing,:)); % Only works for 1 press per finger
                    pressed(fing(1))=1;
                end
            end
        end
        varargout={D};
    case 'Plot_exampletrial'
        % makes a example trial plot
        cmap ={'r','m','b','c','g','k'};
        D=varargin{1};
        inp = varargin{2};
        data=varargin{3};
        tn = varargin{4};
        t = [1:size(inp{tn},2)];
        subplot(3,1,1)
        ylabel('Input')
        for j = 1:6
            tt = find(inp{tn}(j,:)>0.5);
            if (~isempty(tt))
                plot(tt, j, [cmap{j} '.'],'MarkerSize',20);
            end;
            hold on;
        end
        hold off;
        set(gca, 'YTick', 1:6, 'YTickLabels', {'1','2','3','4','5','Go'})
        axis([1 max(t) 0.5 6.5])
        subplot(3,1,2)
        ylabel('Output')
        for j=1:5
            plot(t, data{tn}{3}(j,:),cmap{j},'LineWidth',2);
            hold on;
        end;
        hold off;
        set(gca,'XLim',[1 max(t)]);
    case 'RDM_dynamics'
        % Generate a time resolved RDM
        D=varargin{1};
        data = varargin{2};
        stats = 'G'; % 'D','G' Distance or second moment?
        type = 1; % 1:latent firing, 3: output firing, 5: latent activity
        timepoints = [2:10:250]; % What timepoints in the simulation
        cscale  = [0 10];
        doplot = 1;
        vararginoptions(varargin(3:end),{'stats','type','timepoints','cscale','doplot'});
        
        % Get data
        K = size(D.episode,1);
        H = eye(K)-ones(K)/K;
        C=indicatorMatrix('allpairs',[1:K]);
        for i=1:length(data)
            Z(:,:,i) = data{i}{type};
        end;
        
        % Generate the plot
        for i=1:length(timepoints)
            Zslice=squeeze(Z(:,timepoints(i),:))';
            switch (stats)
                case 'D'
                    Diff = C*Zslice;
                    G(:,:,i)=squareform(sum(Diff.*Diff,2));
                case 'G'
                    G(:,:,i)=H*Zslice*Zslice'*H';
            end;
            if (doplot)
                subplot(5,5,i);
                imagesc(G(:,:,i),cscale);
                title(sprintf('%d',timepoints(i)));
                drawline([24.5:24:100],'dir','vert');
                drawline([24.5:24:100],'dir','horz');
            end;
        end;
        varargout={G};
    case 'RDM_models'
        D=varargin{1};
        features ={'fingers','transitions'}; % Combination of 'fingers','transitions','sequence'
        K = size(D.episode,1);
        H = eye(K)-ones(K)/K;
        C=indicatorMatrix('allpairs',[1:K]);
        stats = 'G'; % 'D' or 'G'
        vararginoptions(varargin(3:end),{'stats','features'});
        cmap= [1 0 0;0.7 0 0.7;0 0 1;0 0.7 0.7;0 0.7 0;0.5 0.5 0.5];
        
        G=[];
        Z={};
        CAT.color={};
        CAT.linestyle={};
        j=1;
        for i=1:length(features)
            switch(features{i})
                case 'fingers'
                    for f=1:5
                        Z{j}=indicatorMatrix('identity',D.targs(:,f));
                        CAT.color{j}=cmap(f,:);
                        CAT.linestyle{j}='-' ;
                        j=j+1;
                    end
                case 'transitions'
                    for f=1:4
                        trans=[D.targs(:,f) D.targs(:,f+1)];
                        [~,~,transID]=unique(trans,'rows');
                        Z{j}=indicatorMatrix('identity',transID);
                        CAT.color{j}=cmap(f,:);
                        CAT.linestyle{j}='--' ;
                        j=j+1;
                    end
                case 'sequence'
                    Z{j}=eye(K);
                    CAT.color{j}=cmap(6,:);
                    CAT.linestyle{j}='-' ;
                    j=j+1;
            end
        end
        for i=1:length(Z)
            switch(stats)
                case 'D'
                    Diff = C*Z{i};
                    G(:,:,i)=squareform(sum(Diff.*Diff,2));
                case 'G'
                    G(:,:,i)=H*Z{i}*Z{i}'*H';
            end;
            subplot(ceil(length(Z)/5),5,i);
            imagesc(G(:,:,i));
            drawline([24.5:24:100],'dir','vert');
            drawline([24.5:24:100],'dir','horz');
        end;
        varargout={G,CAT};
    case 'RDM_regression'
        % Color maps for fingers
        type = 1; % 1:latent firing, 3: output firing, 5: latent activity
        timepoints = [5:2:250];
        features={'fingers'};
        
        vararginoptions(varargin(3:end),{'features','timepoints'});
        
        D=varargin{1};
        data=varargin{2};
        Gemp = SeqTask('RDM_dynamics',D,data,'stats','G','type',type,'timepoints',timepoints,'doplot',0);
        figure(1);
        [Gmod,CAT] = SeqTask('RDM_models',D,data,'stats','G','features',features);
        for i=1:size(Gmod,3)
            x=Gmod(:,:,i);
            X(:,i)=x(:);
        end;
        
        % Do nonneg regression at each time and calculate FSS
        for t=1:length(timepoints)
            y=Gemp(:,:,t);
            y=y(:);
            beta(:,t)=lsqnonneg(X,y);
            tss(t) = y'*y;
            fss(:,t)=sum(bsxfun(@times,X,beta(:,t)').^2,1);
            FSS = sum((X*beta).^2);
        end;
        
        % Plot the fitted and total sums of squares
        figure(2);
        for i=1:size(Gmod,3)
            plot(timepoints,fss(i,:)./tss,'Color',CAT.color{i},'LineWidth',2,'LineStyle',CAT.linestyle{i});
            hold on;
        end;
        % plot(timepoints,tss,'Color',[0 0 0],'LineWidth',1,'LineStyle','-');
        plot(timepoints,FSS./tss,'Color',[0 0 0],'LineWidth',1,'LineStyle',':');
        
        drawline([D.cue(1,:) D.gocue(1,:) mean(D.pressMax)]);
        hold off;
        xlabel('Time');
        ylabel('Proportion variance explained');
    case 'State_Space_Plot'
        % Makes a state-space plot of neuronal trajectories in a specified
        % time window
        % SeqTask('State_Space_Plot',D,data,'timeRange',[160:250])
        D=varargin{1};
        data = varargin{2};
        type =1; % 1: activity 3:output 5:
        K= size(D.episode,1);
        T= size(data{1}{1},2);
        dimensions = 3;
        timeRange = [1:T-20];
        
        % Color maps for fingers
        cmap= [1 0 0;0.7 0 0.7;0 0 1;0 0.7 0.7;0 0.7 0];
        
        % Set the time symbols
        stampTime = [1 D.cue(1,:) D.cueend(1) D.gocue(1) round(mean(D.pressMax)) max(timeRange)];
        stampSymbol = {'^','+','+','+','+','+','o','o','x','x','x','x','x','o'};
        stampColor = [0 0 0;cmap;0.7 0.7 0.7;0 1 0;cmap;0 0 0];
        stampName = {'start','D1','D2','D3','D4','D5','cueend','Go','P1','P2','P3','P4','P5','end'};
        
        vararginoptions(varargin(3:end));
        
        % Adjust the timing symbols to current time window
        stampTime(1) = timeRange(1);
        stampTime(end) = timeRange(end);
        stampTime = stampTime-timeRange(1)+1;
        stampTime(stampTime>stampTime(end))=NaN;
        stampTime(stampTime<1)=NaN;
        
        % Build color map for trials
        cmapTrial = zeros(K,3,5);
        for i = 1:K
            for tar = 1:5
                if D.noGo(i) == 1
                    cmapTrial(i,:,tar) = [0.6 0.6 0.6];
                else
                    cmapTrial(i,:,tar) = cmap(D.targs(i,tar),:);
                end
            end
        end
        
        % Condense data
        for i=1:length(data)
            Z(:,:,i) = data{i}{type};
        end;
        
        % Do the PCA
        pcData = Z(:,timeRange,:);
        pcData = pcData(:,:)';
        pcData = bsxfun(@minus,pcData,mean(pcData));
        
        [coeff, score, eigenval] = pca(pcData);
        vData = eigenval / sum(eigenval);
        score = pcData * coeff(:,1:3);
        
        pc = reshape(score', [3 length(timeRange) K]);
        
        % Plot the trajectories and the markers
        for cond = 1:size(pc,3)
            go = -timeRange(1)+1;
            endCue = D.cueend(1)-timeRange(1)+1;
            endTrial = length(timeRange);
            if dimensions == 3
                plot3(pc(1,:,cond), pc(2,:,cond), pc(3,:,cond), 'Color', cmapTrial(cond,:,1));
                hold on;
                % Generate the stamp symbols
                for i=1:length(stampTime)
                    if (~isnan(stampTime(i)))
                        h(i)=plot3(pc(1,stampTime(i),cond), pc(2,stampTime(i),cond), pc(3,stampTime(i),cond), stampSymbol{i}, 'Color', stampColor(i,:), 'MarkerSize', 5);
                    end;
                end;
            elseif dimensions == 2
                % Todo
            end
        end
        indx = ~isnan(stampTime);
        legend(h(indx),stampName(indx));
        hold off;
    case 'State_Space_movement' 
        % Analyze the state-space trajectories with peak forces 
        D=varargin{1};
        data = varargin{2};
        type =1; % 1: activity 3:output 5:
        K= size(D.episode,1);
        timeRange = round(mean([D.pressMax(:,1)-14 D.pressMax D.pressMax(:,5)+14]));
        contrast = 'all';
        % Color maps for fingers
        cmap= [1 0 0;0.7 0 0.7;0 0 1;0 0.7 0.7;0 0.7 0];
        
        vararginoptions(varargin(3:end));
        
        % Condense data
        for i=1:length(data)
            n=length(timeRange); % Length of a single trial 
            in = [1:n]+(i-1)*n; 
            Z(in,:) = data{i}{type}(:,timeRange)';
            T.Press(in,1) = [0 D.press(i,:) 0];
            T.Prev(in,1) = [0 0 D.press(i,:)]; 
            T.Next(in,1) = [D.press(i,:) 0 0]; 
            T.pressNum(in,1) = [0:6]; 
            T.trial(in,1) = ones(1,n)*i;  
        end
        Z = bsxfun(@minus,Z,mean(Z));
        
        % Maximize the eigenvalues to maximize a certain contrast 
        N=size(T.trial,1); 
        switch (contrast)
            case 'all' 
                contrast= eye(N); 
            case 'currentFinger' 
                contrast=indicatorMatrix('identity_p',T.Press); 
            case 'trialPhase'
                contrast=indicatorMatrix('identity',T.pressNum); 
        end; 
        H = contrast*pinv(contrast);  % Projection matrix 
        [V,L]=eig(conj(Z)'*H'*H*Z); 
        [l,i]   = sort(real(diag(L)),1,'descend');           % Sort the eigenvalues
        V       = V(:,i); 
        score   = Z*V(:,1:3); 
        pc = reshape(score', [3 n K]);
        
        % Plot the trajectories and the markers
        for cond = 1:K
            plot3(pc(1,:,cond), pc(2,:,cond), pc(3,:,cond),'Color',[0.5 0.5 0.5]);
            hold on;
        end
        for i=[1:5]
            indx = find(T.Press==i); 
            h=plot3(score(indx,1), score(indx,2), score(indx,3),'o','MarkerFaceColor',cmap(i,:),'Color',cmap(i,:));
        end; 
        hold off;

    case 'motorSpace'
        net=varargin{1};
end