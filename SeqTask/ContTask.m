function varargout=ContTask(what,varargin)
% Different functions for the evaluation of
% Different neuronal networks
baseDir = '/Users/jdiedrichsen/Dropbox (Diedrichsenlab)/projects/RNN_sequence';
% baseDir = '/srv/diedrichsen/RNN_sequence'; 

switch(what)
    case 'Run_all'
        % Train a network
        simparams=ContTask('Get_simparamsTrain','Cont_M1rr'); % Get parameters
        SeqTask('trainNetwork',simparams); % Train a network
        simparams=SeqTask('Get_simparamsTrain','Gaussian3Flex'); % Get parameters
        SeqTask('trainNetwork',simparams); % Train a network
        
        % Now test it on single trials
        simparams=ContTask('Get_simparamsTest','memorySpan',1);
        [D,v_inputtrain_T,m_targettrain_T]=ContTask('Get_testEpisodes',simparams);
        net = SeqTask('Get_network','Cont_M1b');
        data = SeqTask('Run_simulation',net,v_inputtrain_T); % Run the simulation of the current network
        data = SeqTask('supplement_data',data,D,net);
        save Cont_M1b_1 D data v_inputtrain_T m_targettrain_T net simparams
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
            case 'Cont_M1r'
                simparams.wc = 10; % cost on square of input and output weights
                simparams.firingrate = 0; % cost on firing rate
                simparams.memorySpan = 1;
            case 'Cont_M1rr'
                simparams.wc = 1; % cost on square of input and output weights
                simparams.firingrate = 1; % cost on firing rate
                simparams.memorySpan = 1;

        
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
        D.prev = [NaN;D.target(1:end-1)]; 
        D.prev(D.targetNum==1)=NaN; 
        D.next= [D.target(2:end);NaN]; 
        D.next(D.targetNum==simparams.numTargets)=NaN; 
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
        timepoints = []; % What timepoints in the simulation
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
                lineP=find(diff(D.target(:,1))); %
                drawline(lineP+0.5,'dir','vert');
                drawline(lineP+0.5,'dir','horz');
            end;
        end;
        varargout={G};
    case 'RDM_models'
        D=varargin{1};
        features ={'target','prev','next','prevT','nextT'}; % Combination of 'fingers','transitions','sequence'
        K = size(D.episode,1);
        H = eye(K)-ones(K)/K;
        C=indicatorMatrix('allpairs',[1:K]);
        stats = 'G'; % 'D' or 'G'
        vararginoptions(varargin(2:end),{'stats','features'});
        cmap= [1 0 0;0.3 0 0.3;0 0 1;0 1 1;0 0.4 0;0.5 0.5 0.5];
        
        G=[];
        Z={};
        CAT.color={};
        CAT.linestyle={};
        j=1;
        for i=1:length(features)
            switch(features{i})
                case 'target'
                    Z{j}=indicatorMatrix('identity',D.target);
                    CAT.color{j}=[0 0 0];
                    CAT.linestyle{j}='-' ;
                    CAT.leg{j}='Target';
                    j=j+1;
                case 'prev'
                    Z{j}=indicatorMatrix('identity',D.prev);
                    CAT.color{j}=[1 0 0];
                    CAT.linestyle{j}='-' ;
                    CAT.leg{j}='Previous';
                    j=j+1;
                case 'next'
                    Z{j}=indicatorMatrix('identity',D.next);
                    CAT.color{j}=[0 0 1];
                    CAT.linestyle{j}='-' ;
                    CAT.leg{j}='Next';
                    j=j+1;
                case 'next2'
                    Z{j}=indicatorMatrix('identity',D.memoryGo(:,3));
                    CAT.color{j}=[0 0.5 0.5];
                    CAT.linestyle{j}='-' ;
                    CAT.leg{j}='Next2';
                    j=j+1;
                case 'prevT'
                    trans=[D.prev D.target];
                    [~,~,transID]=unique(trans,'rows');
                    Z{j}=indicatorMatrix('identity',transID);
                    CAT.color{j}=[1 0 0];
                    CAT.linestyle{j}='--' ;
                    CAT.leg{j}='Prev Trans';
                    j=j+1;
                case 'nextT'
                    trans=[D.target D.next];
                    [~,~,transID]=unique(trans,'rows');
                    Z{j}=indicatorMatrix('identity',transID);
                    CAT.color{j}=[0 0 1];
                    CAT.linestyle{j}='--';
                    CAT.leg{j}='Next Trans';
                    j=j+1;
                otherwise 
                    error('unknown model'); 
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
            lineP=find(diff(D.target)); %
            drawline(lineP+0.5,'dir','vert');
            drawline(lineP+0.5,'dir','horz');
        end;
        varargout={G,CAT,Z};
    case 'RDM_regression'
        % Color maps for fingers
        type = 1; % 1:latent firing, 3: output firing, 5: latent activity
        features ={'target','prev','next','prevT','nextT'}; % Combination of 'fingers','transitions','sequence'
        normalize=0;
        units=[];
        targetNum = 4; 
        timepoints = []; 
        D=varargin{1};
        data=varargin{2};

        vararginoptions(varargin(3:end),{'type','features','targetNum',...
            'normalize','units','timepoints'});
        
        % Concentrate on one target and specific times 
        D=getrow(D,D.targetNum==targetNum); 
        if (isempty(timepoints)); 
            timepoints=[D.goOnset(1)-10:5:D.end(1)];
        end; 
    
        
        Gemp = ContTask('RDM_dynamics',D,data,'stats','G','type',type,...
            'timepoints',timepoints,'units',units,'doplot',0);
        figure(1);
        [Gmod,CAT] = ContTask('RDM_models',D,'stats','G','features',features);
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
        drawline([D.goOnset(1) D.peakPress(1) D.endPress(1)]);
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
        cmap= [1 0 0;0.7 0 0.7;0 0 0.7;0 0.8 0.8;0 0.3 0];

        
        % Set the time symbols
        stampTime = [D.cueOnset D.goOnset D.peakPress]-1;
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
            stamps=stampTime(D.episode==cond,:); 
            plot3(pc(1,1,cond), pc(2,1,cond), pc(3,1,cond), '^', 'Color', 'k', 'MarkerSize', 5);
            for i=1:3 
                plot3(pc(1,stamps(:,i),cond), pc(2,stamps(:,i),cond), pc(3,stamps(:,i),cond), stampSymbol{i}, 'Color', 'k', 'MarkerSize', 5);
            end; 
        end
        hold off;
    case 'State_Space_subspace' 
        % Decomposes the data around a specific time range based on 
        % on the specific RDM models, defined by a specific time point 
        D=varargin{1};
        data = varargin{2};
        
        timeRange = []; 
        targetNum = 4; 
        targets = [1:5]; 
        [numUnits,T]= size(data{1}{1});
        units = [1:numUnits];
        type =1; % 1: activity 3:output 5:
        dimensions = 3;
        removeMean = 0; 
        subFeature = {'next2','next','target'}; % Features that define the subspace 
        subTimestamp = [1 1 2];  
        D.next2 = D.memoryGo(:,3);
        
        vararginoptions(varargin(3:end),{'type','timeRange','units',...
            'targetNum','targets','removeMean','subFeature','subTimestamp'});

        numSubS = length(subFeature); % Number of subfeatures 

        % Concentrate on one target and specific times 
        D=getrow(D,D.targetNum==targetNum); 
        
        % Optional visualize a few targets only: 
        if (any(strcmp(subFeature,'next2')))
            indx = find(ismember(D.target,targets) & ismember(D.next,targets) & ismember(D.next2,targets)); 
        else
            indx = find(ismember(D.target,targets) & ismember(D.next,targets) );        
        end; 
        
        % indx = [1:length(D.target)]'; 
        numCond = length(indx);
        D=getrow(D,indx); 
        
        if (isempty(timeRange)) 
            timeRange=[D.goOnset(1)-10:1:D.end(1)-1];
        end 

        [Gmod,CAT,F] = ContTask('RDM_models',D,'stats','G','features',subFeature);
        % Color maps for fingers
        cmap= [1 0 0;0.7 0 0.7;0 0 0.7;0 0.8 0.8;0 0.3 0];
        
        % Set the time symbols
        stampTime = [D.goOnset(1) D.peakPress(1)-12];
        stampSymbol = {'^','o'};
        stampName = {'cue','go'};
        
        % Condense data
        for i=1:length(indx)
            if (type ==3 || type==6) 
                Z(:,:,i) = data{indx(i)}{type};
            else 
                Z(:,:,i) = data{indx(i)}{type}(units,:);
            end; 
        end;
        
        % All Data 
        pcData = Z(:,timeRange,:);
        pcData = pcData(:,:)';
        pcData = bsxfun(@minus,pcData,mean(pcData));
        
        % Determine subspaces on the feature / time frame that is important
        for i=1:numSubS 
            % Define contrast subspace 
            if (removeMean) 
                F{i}=bsxfun(@minus,F{i},mean(F{i}));
            end; 
            P = F{i}*pinv(F{i});  % Projection matrix 
            Y = squeeze(Z(:,stampTime(subTimestamp(i))-1,:))'; 
            % Check if RDM looks correct at this place 
            % H=eye(25)-ones(25)/25; 
            % imagesc(H*Y*Y'*H'); 
            [v,L]=eig(Y'*P'*P*Y); 
            [l,j]   = sort(real(diag(L)),1,'descend');           % Sort the eigenvalues
            V{i}       = v(:,j(1:dimensions)); 
            lambda{i}  = l(j(1:dimensions)); 
            score{i}       = pcData*V{i}; 
            pc{i} = reshape(score{i}', [dimensions length(timeRange) numCond]);
        
            subplot(1,numSubS,i); 
            for cond = 1:size(pc{i},3)
                plot3(pc{i}(1,:,cond), pc{i}(2,:,cond), pc{i}(3,:,cond), 'Color',  cmap(D.(subFeature{i})(cond),:));
                % axis equal; 
                hold on;
                % Generate the stamp symbols
                stamps=stampTime-timeRange(1); 
                % plot3(pc{i}(1,1,cond), pc{i}(2,1,cond), pc{i}(3,1,cond), ...
                %     '^', 'Color', cmap(D.target(cond),:), 'MarkerSize', 5);
                plot3(pc{i}(1,end,cond), pc{i}(2,end,cond), pc{i}(3,end,cond), ...
                    'x', 'Color',  cmap(D.(subFeature{i})(cond),:), 'MarkerSize', 5);
                for j=1:length(stamps) 
                    plot3(pc{i}(1,stamps(j),cond), pc{i}(2,stamps(j),cond), pc{i}(3,stamps(j),cond), ...
                        stampSymbol{j}, 'Color', cmap(D.(subFeature{i})(cond),:), 'MarkerSize', 5);
                end
            end
            lims = axis; 
            spacing=(lims(2)-lims(1))/8; 
            set(gca,'XTick',[lims(1):spacing:lims(2)],'YTick',[lims(3):spacing:lims(4)],'ZTick',[lims(5):spacing:lims(6)]); 
            set(gca,'XGrid','on','YGrid','on','ZGrid','on'); 
            set(gca,'XTickLabel',{},'YTickLabel',{},'ZTickLabel',{}); 
            xlabel('D1');ylabel('D2');zlabel('D3'); 
            
        end
        hold off;
        varargout={V,lambda,pc}; 
    case 'subspaceAlignment' 
        V=varargin{1}; 
        lambda = varargin{2}; 
        Vw=[]; 
        for i=1:length(V)
            Vw{i}=bsxfun(@times,V{i},sqrt(lambda{i}'));
        end
        for i=1:length(V)
            for j=1:length(V)
                C = (Vw{i}'*Vw{j}); 
                HSIK(i,j)=sum(sum(C.^2)); 
            end
        end 
        CKA = corrcov(HSIK);
        keyboard;             
    case 'Figure_Sub2' % Subspace Figure for the Grant 
        load Cont_M2_2.mat; 
        spacing=0.15; 
        subFeature = {'next','target'}; % Features that define the subspace 
        subTimestamp = [ 1 2];  
        ContTask('State_Space_subspace',D,data,'targets',[1 2 5],...
            'subFeature',subFeature,'subTimestamp',subTimestamp); 
        set(gcf,'PaperPosition',[2 2 8 3]);wysiwyg; 
        subplot(1,2,1); 
        view(100,10); 
        axis equal; 
        lims = axis; 
        set(gca,'XTick',[lims(1):spacing:lims(2)],'YTick',[lims(3):spacing:lims(4)],'ZTick',[lims(5):spacing:lims(6)]); 
        set(gca,'XGrid','on','YGrid','on','ZGrid','on'); 
        set(gca,'XTickLabel',{},'YTickLabel',{},'ZTickLabel',{}); 
        subplot(1,2,2);
        view(-75,10); 
        axis equal; 
        lims = axis; 
        set(gca,'XTick',[lims(1):spacing:lims(2)],'YTick',[lims(3):spacing:lims(4)],'ZTick',[lims(5):spacing:lims(6)]); 
        set(gca,'XGrid','on','YGrid','on','ZGrid','on'); 
        set(gca,'XTickLabel',{},'YTickLabel',{},'ZTickLabel',{}); 
    case 'Figure_Sub3' % Subspace Figure for the Grant 
        load Cont_M3_3.mat; 
        spacing=0.15; 
        subFeature = {'next2','next','target'}; % Features that define the subspace 
        subTimestamp = [1 1 2];  
        ContTask('State_Space_subspace',D,data,'targets',[1 2 5],...
            'subFeature',subFeature,'subTimestamp',subTimestamp,'removeMean',1); 
        set(gcf,'PaperPosition',[2 2 12 4]);wysiwyg; 
        views=[[-195 25];[-50 28];[125,23]]; 
        for i=1:3 
            subplot(1,3,i); 
            view(views(i,1),views(i,2)); 
            % axis equal; 
            lims = axis; 
            set(gca,'XTick',[lims(1):spacing:lims(2)],'YTick',[lims(3):spacing:lims(4)],'ZTick',[lims(5):spacing:lims(6)]); 
            set(gca,'XGrid','on','YGrid','on','ZGrid','on'); 
            set(gca,'XTickLabel',{},'YTickLabel',{},'ZTickLabel',{}); 
        end 
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