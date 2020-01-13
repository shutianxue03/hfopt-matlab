function varargout=ContTask(what,varargin)
% Different functions for the evaluation of
% Different neuronal networks
baseDir = '/Users/jdiedrichsen/Dropbox (Diedrichsenlab)/projects/RNN_sequence';

switch(what)
    case 'Run_all'
        % Train a network
        simparams=SeqTask('Get_simparamsTrain','Gaussian3Rep'); % Get parameters
        SeqTask('trainNetwork',simparams); % Train a network
        simparams=SeqTask('Get_simparamsTrain','Gaussian3Flex'); % Get parameters
        SeqTask('trainNetwork',simparams); % Train a network
        
        % Now test it on single trials
        simparams=SeqTask('Get_simparamsTest','Gaussian3Rep');
        [D,v_inputtrain_T,m_targettrain_T]=SeqTask('Get_singleTrialTest',simparams);
        net = SeqTask('Get_network','Gaussian3Rep160');
        data = SeqTask('Run_simulation',net,v_inputtrain_T); % Run the simulation of the current network
        D=SeqTask('performance',data,D);
        data = SeqTask('supplement_data',data,D,net);
    case 'Get_simparamsTrain'
        
        simparams.name = varargin{1};
        % How often to save a checkpoint of training
        simparams.saveEvery = 5;
        % Which task to run
        simparams.taskfun = @ContTask_taskfun; % or @SeqTask_taskfun_threshold
        % Plotting function during training
        simparams.plotfun = @ContTask_hfopt_Plot;
        
        simparams.numEpisodes = 20;       % Number of Episodes to simulation
        simparams.numTargets = 15;      % Number of elements in the sequence
        simparams.targetset = [1:3];    % What are the possible targets?
        simparams.preTime = 10; % Time before visual cues
        simparams.cueDur = 10; % How long is each cue on?
        simparams.cue2go = [10 40];  % Range of cue-2-go period 
        simparams.RT = 12;         % From onset of go-cue to beginning of force production
        simparams.forceWidth= 25; % How long is each force press?
        simparams.press2cue = [10 40]; % end of press to onset of new cue
        simparams.memoryspan = 3; 
        
        % Network size parameters
        simparams.N = 160; % Number of neurons
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
        
        switch (simparams.name)
            case 'Cont_M1'
                simparams.memoryspan = 3; 
            case 'Cont_M2'
                simparams.memoryspan = 3; 
        end
        varargout={simparams};
    case 'Get_simparamsTest'
        % Generates simulation parameters for single trial tyes
        type=varargin{1}; 
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
        simparams.forceIPI= 10;   % How long between onsets of each press?
        simparams.RT = 12;         % From onset of go-cue to beginning of force production
        simparams.moveTime = (simparams.forceWidth+simparams.forceIPI)*simparams.numTargets+20; % Total movement time
        simparams.noGoFreq=0;
        switch (type) 
            case 'Gaussian5NoRep'
                simparams.numTargets=5;
                simparams.targetset = [1:5];   % What are the possible targets?
                simparams.withreplace = false;   % Targets can repeat?
            case 'Gaussian3Rep' 
                simparams.numTargets=5;
                simparams.targetset = [1:3];   % What are the possible targets?
                simparams.withreplace = true;   % Targets can repeat?                
        end
        varargout={simparams};
    case 'Get_singleTrialTest'
        % Get an exhaustive set of trials
        simparams=varargin{1};
        
        if (simparams.withreplace) 
            K=length(simparams.targetset); 
            D.targs(1:K,1)=simparams.targetset; 
            for i=1:simparams.numTargets-1
                N=size(D.targs,1); 
                D.targs=[kron(simparams.targetset',ones(N,1)) repmat(D.targs,K,1)]; 
            end
        else 
            D.targs=perms(simparams.targetset);
        end 
        N=size(D.targs,1);
        D.episode = [1:N]';
        D.trial = ones(N,1);
        D.noGo = zeros(N,1);
        D.memLength = ones(N,1)*simparams.memPeriod(1);
        [v_inputtrain_T,D]=SeqTask_taskfun_inputs(D,simparams);
        m_targettrain_T=SeqTask_taskfun_outputs(D,simparams);
        varargout={D,v_inputtrain_T,m_targettrain_T};
    case 'Plot_Learningcurve'
        networks = {'GaussianNoRep1','GaussianNoRep2','GaussianTest'};
        vararginoptions(varargin,{'networks'});
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
    case 'supplement_data'
        % Calculates some supplements on the data
        % Data{episode}{1} = Firing of latent units across time (after nonlinearity)
        % Data{episode}{3} = Firing rate of output units
        % Data{episode}{4} = Firing of latent units - weighted by output weights
        % Data{episode}{5} = Activity of latent units (before nonlinearity)
        % Data{episode}{6} = Activity of output units (before nonlinearity)
        data = varargin{1};
        D    = varargin{2};
        net  = varargin{3};
        N=numel(data);
        [n_Wru_v, n_Wrr_n, m_Wzr_n, n_x0_c, n_bx_1, m_bz_1] = unpackRNN(net, net.theta);
        % Output weighting:
        outW=sqrt(sum(m_Wzr_n.^2,1));
        
        for n=1:N
            data{n}{4}=bsxfun(@times,outW',data{n}{1});
            data{n}{6}=bsxfun(@plus,m_Wzr_n*data{n}{1},m_bz_1);
        end;
        varargout={data};
        
    case 'performance'
        % Rate performance and establish timing
        % D=SeqTask('performance',data,D);
        data = varargin{1};
        D= varargin{2};
        thresh =0.4;
        for i=1:length(data)
            out=data{i}{3};
            press = 0;
            for t=D.gocue(i):size(out,2)
                fing = find(out(:,t-1)<thresh & out(:,t)>thresh);
                if ~isempty(fing)
                    press = press+1;
                    D.press(i,press)=fing(1);
                    D.pressOnset(i,press)=t;
                    [~,x]=max(out(fing,t:t+8)); % Only works for 1 press per finger
                    D.pressMax(i,press)=t+x-1; 
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
        tn = 1;
        type =1;
        vararginoptions(varargin(4:end),{'tn','type'});
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
        subplot(3,1,3)
        if (size(data{tn}{type},1)==5)
            for j=1:5
                plot(t, data{tn}{type}(j,:),cmap{j},'LineWidth',2);
                hold on;
            end;
            hold off;
        else
            plot(t, data{tn}{type});
        end;
        set(gca,'XLim',[1 max(t)]);
    case 'RDM_dynamics'
        % Generate a time resolved RDM
        D=varargin{1};
        data = varargin{2};
        stats = 'G'; % 'D','G' Distance or second moment?
        type = 1; % 1:latent firing, 3: output firing, 5: latent activity
        timepoints = [2:10:250]; % What timepoints in the simulation
        units = []; 
        cscale  = [0 10];
        doplot = 1;
        vararginoptions(varargin(3:end),{'stats','type','timepoints',...
            'cscale','doplot','units'});
        
        % Get data
        K = size(D.episode,1);
        H = eye(K)-ones(K)/K;
        C=indicatorMatrix('allpairs',[1:K]);
        [numUnits,numTimes]=size(data{1}{type});
        if (isempty(units))
            units = 1:numUnits; 
        end; 
        for i=1:K
            Z(:,:,i) = data{i}{type}(units,:);
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
                lineP=find(diff(D.targs(:,1))); % 
                drawline(lineP+0.5,'dir','vert');
                drawline(lineP+0.5,'dir','horz');
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
        cmap= [1 0 0;0.3 0 0.3;0 0 1;0 1 1;0 0.4 0;0.5 0.5 0.5];
        
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
                        CAT.leg{j}=sprintf('f%d',f);
                        j=j+1;
                    end
                case 'transitions'
                    for f=1:4
                        trans=[D.targs(:,f) D.targs(:,f+1)];
                        [~,~,transID]=unique(trans,'rows');
                        Z{j}=indicatorMatrix('identity',transID);
                        CAT.color{j}=cmap(f,:);
                        CAT.linestyle{j}='--' ;
                        CAT.leg{j}=sprintf('t%d',f);
                        j=j+1;
                    end
                case 'endSeq'
                    for f=1:4
                        trans=D.targs(:,f+1:5);
                        [~,~,transID]=unique(trans,'rows');
                        Z{j}=indicatorMatrix('identity',transID);
                        CAT.color{j}=cmap(f,:);
                        CAT.linestyle{j}='--' ;
                        CAT.leg{j}=sprintf('t%d',f);
                        j=j+1;
                    end
                case 'startSeq'
                    for f=1:4
                        trans=D.targs(:,1:f);
                        [~,~,transID]=unique(trans,'rows');
                        Z{j}=indicatorMatrix('identity',transID);
                        CAT.color{j}=cmap(f,:);
                        CAT.linestyle{j}='--' ;
                        CAT.leg{j}=sprintf('t%d',f);
                        j=j+1;
                    end
                case 'sequence'
                    Z{j}=eye(K);
                    CAT.color{j}=cmap(6,:);
                    CAT.linestyle{j}=':' ;
                    CAT.leg{j}='seq';
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
                    G(:,:,i)=G(:,:,i)./trace(G(:,:,i));
            end;
            subplot(ceil(length(Z)/5),5,i);
            imagesc(G(:,:,i));
            lineP=find(diff(D.targs(:,1))); % 
            drawline(lineP+0.5,'dir','vert');
            drawline(lineP+0.5,'dir','horz');
        end;
        varargout={G,CAT};
    case 'RDM_regression'
        % Color maps for fingers
        type = 1; % 1:latent firing, 3: output firing, 5: latent activity
        timepoints = [5:2:250];
        features={'fingers'};
        normalize=0; 
        units=[]; 
        vararginoptions(varargin(3:end),{'type','features','timepoints',...
            'normalize','units'});
        
        D=varargin{1};
        data=varargin{2};
        Gemp = SeqTask('RDM_dynamics',D,data,'stats','G','type',type,...
            'timepoints',timepoints,'units',units,'doplot',0);
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
            % These are sums of squares of all the entries of the G-matrix
            % tss(t) = y'*y;
            % fss(:,t)=sum(bsxfun(@times,X,beta(:,t)').^2,1);
            % FSS = sum((X*beta).^2);
            %
            % Here we use only diagnonal of the G-matrix: Pattern variance
            tss(t)=trace(Gemp(:,:,t));
            for i=1:size(Gmod,3);
                fss(i,t)=trace(Gmod(:,:,i))*beta(i,t);
            end;
        end;
        FSS = sum(fss,1);
        % Plot the fitted and total sums of squares
        if (normalize)
            fss=bsxfun(@rdivide,fss,FSS); 
        end
        figure(2);
        for i=1:size(Gmod,3)
            graphEl(i)=plot(timepoints,fss(i,:),'Color',CAT.color{i},'LineWidth',3,'LineStyle',CAT.linestyle{i});
            hold on;
        end
        if (~normalize) 
            plot(timepoints,FSS,'Color',[1 0 0],'LineWidth',1,'LineStyle',':');
            plot(timepoints,tss,'Color',[0 0 0],'LineWidth',1,'LineStyle',':');
        end;         
        drawline([D.cue(1,:) D.gocue(1,:) mean(D.pressMax)]);
        hold off;
        set(gca,'XLim',[min(timepoints) max(timepoints)]); 
        legend(graphEl,CAT.leg);
        xlabel('Time');
        ylabel('Proportion variance explained');
    case 'State_Space_Plot'
        % Makes a state-space plot of neuronal trajectories in a specified
        % time window
        % SeqTask('State_Space_Plot',D,data,'timeRange',[160:250])
        D=varargin{1};
        data = varargin{2};

        K= size(D.episode,1);
        [numUnits,T]= size(data{1}{1},2);
        units = [1:numUnits]; 
        type =1; % 1: activity 3:output 5:
        dimensions = 3;
        timeRange = [1:T-20];
        colorByPress = 1; % Determine color of line by press
        % Color maps for fingers
        cmap= [1 0 0;0.7 0 0.7;0 0 1;0 0.7 0.7;0 0.7 0];
        
        % Set the time symbols
        stampTime = [1 D.cue(1,:) D.cueend(1) D.gocue(1) round(mean(D.pressMax)) max(timeRange)];
        stampSymbol = {'^','+','+','+','+','+','o','o','x','x','x','x','x','o'};
        stampColor = [0 0 0;cmap;0.7 0.7 0.7;0 1 0;cmap;0 0 0];
        stampName = {'start','D1','D2','D3','D4','D5','cueend','Go','P1','P2','P3','P4','P5','end'};
        
        vararginoptions(varargin(3:end),{'type','timeRange','colorByPress','units'});
        
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
            Z(:,:,i) = data{i}{type}(units,:);
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
                plot3(pc(1,:,cond), pc(2,:,cond), pc(3,:,cond), 'Color', cmapTrial(cond,:,colorByPress));
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
    case 'Movement_data'
        % Analyze the state-space trajectories with peak forces
        D=varargin{1};
        data = varargin{2};
        type =1; % 1: activity 3:output 5:
        K= size(D.episode,1);
        timeRange = round(mean([D.pressMax(:,1)-14 D.pressMax D.pressMax(:,5)+14]));
        [numUnits,~]= size(data{1}{type});
        units = [1:numUnits]; 

        vararginoptions(varargin(3:end),{'type','units'});
        
        % Condense data
        for i=1:length(data)
            n=length(timeRange); % Length of a single trial
            in = [1:n]+(i-1)*n;
            Z(in,:) = data{i}{type}(units,timeRange)';
            T.Press(in,1) = [0 D.press(i,:) 0];
            T.Prev(in,1) = [0 0 D.press(i,:)];
            T.Next(in,1) = [D.press(i,:) 0 0];
            T.pressNum(in,1) = [0:6];
            T.trial(in,1) = ones(1,n)*i;
        end
        Z = bsxfun(@minus,Z,mean(Z));
        
        % Sort the data by trial phase
        [~,j]=sort(T.pressNum);
        T=getrow(T,j);
        Z=Z(j,:); 
        varargout={T,Z};
    case 'Movement_state_space'
        % [T,Z]=SeqTask('Movement_data',D,data);
        % SeqTask('Movement_state_space',T,Z);
        T=varargin{1};
        Z=varargin{2};
        contrast = 'all';
        cmap= [1 0 0;0.7 0 0.7;0 0 1;0 0.7 0.7;0 0.7 0];
        vararginoptions(varargin(3:end));
        % Maximize the eigenvalues to maximize a certain contrast
        N=size(T.trial,1);
        K=max(T.trial);
        n=N/K;
        
        
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
            indx = find(T.pressNum==i);
            h=plot3(score(indx,1), score(indx,2), score(indx,3),'o','MarkerFaceColor',cmap(i,:),'Color',cmap(i,:));
        end;
        hold off;
        
    case 'Movement_RDM'
        %figure(1);
        %[T,Z]=SeqTask('Movement_data',D,data,'type',1);
        %G=SeqTask('Movement_RDM',T,Z,'remove','trialPhase');
        %figure(2);
        %[T,Z]=SeqTask('Movement_data',D,data,'type',3);
        %G=SeqTask('Movement_RDM',T,Z,'remove','trialPhase');
        
        T=varargin{1};
        Z=varargin{2};
        remove = 'mean';
        cscale = [0 1]; 
        vararginoptions(varargin(3:end),{'remove','cscale'});
        
        N=size(T.trial,1);
        switch (remove)
            case 'mean'
                C=ones(N,1);
            case 'trialPhase'
                C=indicatorMatrix('identity',T.pressNum);
            case 'none' 
                C=zeros(N,0);
        end;
        R=eye(N)-C*pinv(C);
        Gemp=R*Z*Z'*R';
        imagesc(Gemp,cscale);
        a=find(diff(T.pressNum)~=0);
        drawline(a,'dir','vert');
        drawline(a,'dir','horz');
        varargout={Gemp};
    case 'Movement_RDM_models'
        T=varargin{1};
        features ={'currentF','prevF','nextF'}; % Combination of 'fingers','transitions','sequence'
        remove = 'trialPhase'; 
        N = size(T.trial,1);
        
        stats = 'G'; % 'D' or 'G'
        vararginoptions(varargin(2:end),{'stats','features','remove'});
        cmap= [1 0 0;0.3 0 0.3;0 0 1;0 1 1;0 0.4 0;0.5 0.5 0.5];
            switch (remove)
            case 'mean'
                C=ones(N,1);
            case 'trialPhase'
                C=indicatorMatrix('identity',T.pressNum);
            case 'none' 
                C=zeros(N,0);
        end;
        R=eye(N)-C*pinv(C);
        R=eye(N)-C*pinv(C);
        
        G=[];
        Z={};
        CAT.color={};
        CAT.linestyle={};
        j=1;
        for i=1:length(features)
            switch(features{i})
                case 'currentF'
                    Z{j}=indicatorMatrix('identity_p',T.Press);
                    CAT.color{j}=[0 0 0];
                    CAT.linestyle{j}='-' ;
                    CAT.leg{j}='currentF';
                    j=j+1;
                case 'prevF'
                    Z{j}=indicatorMatrix('identity_p',T.Prev);
                    CAT.color{j}=[1 0 0];
                    CAT.linestyle{j}='-' ;
                    CAT.leg{j}='prevF';
                    j=j+1;
                case 'nextF'
                    Z{j}=indicatorMatrix('identity_p',T.Next);
                    CAT.color{j}=[0 0 1];
                    CAT.linestyle{j}='-' ;
                    CAT.leg{j}='nextF';
                    j=j+1;
                case 'startSeq'
                case 'sequence'
            end
        end
        for i=1:length(Z)
            switch(stats)
                case 'D'
                    Diff = C*Z{i};
                    G(:,:,i)=squareform(sum(Diff.*Diff,2));
                    
                case 'G'
                    G(:,:,i)=R*Z{i}*Z{i}'*R';
                    G(:,:,i)=G(:,:,i)./trace(G(:,:,i));
            end;
            subplot(ceil(length(Z)/5),5,i);
            imagesc(G(:,:,i));
            lineP=find(diff(T.pressNum)); % 
            drawline(lineP+0.5,'dir','vert');
            drawline(lineP+0.5,'dir','horz');
        end;
        varargout={G,CAT};
    case 'Movement_RDM_regression'
        % Color maps for fingers
        features={'currentF','prevF','nextF'};
        normalize=0; 
        units=[]; 
        cscale = [0 2];
        remove='trialPhase';
        vararginoptions(varargin(3:end),{'type','features','timepoints',...
            'normalize','units'});
        
        T=varargin{1};
        Z=varargin{2};

        figure(1); 
        subplot(1,2,1); 
        Gemp=SeqTask('Movement_RDM',T,Z,'remove',remove,'cscale',cscale);
        N=size(Gemp,1); 
        
        figure(2);
        [Gmod,CAT] = SeqTask('Movement_RDM_models',T,'features',features);
        for i=1:size(Gmod,3)
            x=Gmod(:,:,i);
            X(:,i)=x(:);
        end
        
        % Do nonneg regression at each time and calculate FSS
        y=Gemp(:);
        beta=lsqnonneg(X,y);
            % These are sums of squares of all the entries of the G-matrix
            % tss(t) = y'*y;
            % fss(:,t)=sum(bsxfun(@times,X,beta(:,t)').^2,1);
            % FSS = sum((X*beta).^2);
            %
            % Here we use only diagnonal of the G-matrix: Pattern variance
        TSS=trace(Gemp);
        Gpred=reshape(X*beta,N,N);
        % Plot the fitted and total sums of squares
        figure(1); 
        subplot(1,2,2); 
        imagesc(Gpred,cscale); 
        
        keyboard; 

end