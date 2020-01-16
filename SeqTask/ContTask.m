function varargout=ContTask(what,varargin)
% Different functions for the evaluation of
% Different neuronal networks
baseDir = '/Users/jdiedrichsen/Dropbox (Diedrichsenlab)/projects/RNN_sequence';

switch(what)
    case 'Run_all'
        % Train a network
        simparams=ContTask('Get_simparamsTrain','Cont_M1'); % Get parameters
        SeqTask('trainNetwork',simparams); % Train a network
        simparams=SeqTask('Get_simparamsTrain','Gaussian3Flex'); % Get parameters
        SeqTask('trainNetwork',simparams); % Train a network
        
        % Now test it on single trials
        simparams=ContTask('Get_simparamsTest','memorySpan',1);
        [D,v_inputtrain_T,m_targettrain_T]=ContTask('Get_testEpisodes',simparams);
        net = SeqTask('Get_network','Cont_M1');
        data = SeqTask('Run_simulation',net,v_inputtrain_T); % Run the simulation of the current network
        data = SeqTask('supplement_data',data,D,net);
    case 'Get_simparamsTrain'
        simparams.name = varargin{1};
        % How often to save a checkpoint of training
        simparams.saveEvery = 5;
        % Which task to run
        simparams.taskfun = @ContTask_taskfun; % or @SeqTask_taskfun_threshold
        % Plotting function during training
        simparams.plotfun = @SeqTask_hfopt_Plot;
        
        simparams.numEpisodes = 20;       % Number of Episodes to simulation
        simparams.numTargets = 15;      % Number of elements in the sequence
        simparams.targetset = [1:5];    % What are the possible targets?
        simparams.preTime = 10; % Time before visual cues
        simparams.cueDur = 10; % How long is each cue on?
        simparams.postCue = [10 60];  % Range of after cue interval
        simparams.goDur   = 8;
        simparams.RT = 12;         % From onset of go-cue to beginning of force production
        simparams.forceWidth= 25; % How long is each force press?
        simparams.postPress = [10 30]; % end of press to onset of new cue
        simparams.memorySpan = [];
        
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
                simparams.memorySpan = 1;
            case 'Cont_M2'
                simparams.memorySpan = 2;
            case 'Cont_M3'
                simparams.memorySpan = 3;
            case 'Cont_Mv'
                simparams.memorySpan = [1 3];
        end
        varargout={simparams};
    case 'Get_simparamsTest'
        % Generates simulation parameters for single trial tyes
        memorySpan = 1;
        vararginoptions(varargin,{'memorySpan'}); 
        % Which task to run
        simparams.taskfun = @ContTask_taskfun; % or @SeqTask_taskfun_threshold
        % Plotting function during training
        simparams.plotfun = @SeqTask_hfopt_Plot;
        
        simparams.numEpisodes = 5^memorySpan;     % Number of Episodes to simulation
        simparams.numTargets = 3+2*memorySpan;      % Number of elements in the sequence
        simparams.targetset = [1:5];    % What are the possible targets?
        simparams.preTime = 10; % Time before visual cues
        simparams.cueDur = 10; % How long is each cue on?
        simparams.postCue = 40;  % Range of after cue interval
        simparams.goDur   = 8;
        simparams.RT = 12;         % From onset of go-cue to beginning of force production
        simparams.forceWidth= 25; % How long is each force press?
        simparams.postPress = 20; % end of press to onset of new cue
        simparams.memorySpan = memorySpan;
        varargout={simparams};   
    case 'Get_testEpisodes'
        % Get an exhaustive set of epsiodes 
        simparams=varargin{1};
                
        K=length(simparams.targetset); 
        TARGS(1:K,1)=simparams.targetset; 
        for i=1:simparams.memorySpan-1
            N=size(TARGS,1); 
            TARGS=[kron(simparams.targetset',ones(N,1)) repmat(TARGS,K,1)]; 
        end
        n=size(TARGS,1);
        D.episode = kron([1:n]',ones(simparams.numTargets,1));
        N = length(D.episode);  
        D.targetNum = kron(ones(n,1),[1:simparams.numTargets]');
        D.memorySpan = ones(N,1)*simparams.memorySpan;
        % Assign all targets as random 
        D.target(1:N,1) = simparams.targetset(randsample(length(simparams.targetset),N,1));
        % Now insert the target sequence starting from position 4 
        for i=1:simparams.memorySpan
            D.target(D.targetNum==3+i) = TARGS(:,i); % Fill in targets 
        end; 
        
        [v_inputtrain_T,D]=ContTask_taskfun_inputs(D,simparams);
        m_targettrain_T=ContTask_taskfun_outputs(D,simparams);
        varargout={D,v_inputtrain_T,m_targettrain_T};
        
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
        
        K= max(D.episode);
        [numUnits,T]= size(data{1}{1});
        units = [1:numUnits];
        type =1; % 1: activity 3:output 5:
        dimensions = 3;
        timeRange = [1:T];
        % Color maps for fingers
        cmap= [1 0 0;0.7 0 0.7;0 0 1;0 0.7 0.7;0 0.7 0];
        
        % Set the time symbols
        stampTime = [D.cueOnset D.goOnset D.peakPress];
        stampSymbol = {'+','o','x'};
        stampName = {'cue','go','press'};
        
        vararginoptions(varargin(3:end),{'type','timeRange','units'});
        
        % Condense data
        for i=1:length(data)
            if (type ==3 || type==6) 
                Z(:,:,i) = data{i}{type};
            else 
                Z(:,:,i) = data{i}{type}(units,:);
            end; 
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
            plot3(pc(1,:,cond), pc(2,:,cond), pc(3,:,cond), 'Color', 'k');
            hold on;
            % Generate the stamp symbols
            DD=getrow(D,D.episode==cond); 
            plot3(pc(1,1,cond), pc(2,1,cond), pc(3,1,cond), '^', 'Color', 'k', 'MarkerSize', 5);
            plot3(pc(1,DD.cueOnset,cond), pc(2,DD.cueOnset-1,cond), pc(3,DD.cueOnset,cond), stampSymbol{1}, 'Color', 'k', 'MarkerSize', 5);
            plot3(pc(1,DD.goOnset,cond), pc(2,DD.goOnset-1,cond), pc(3,DD.goOnset,cond), stampSymbol{2}, 'Color', 'k', 'MarkerSize', 5);
            plot3(pc(1,DD.peakPress,cond), pc(2,DD.peakPress-1,cond), pc(3,DD.peakPress,cond), stampSymbol{3}, 'Color', 'k', 'MarkerSize', 5);
        end
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
        
end;