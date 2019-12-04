function varargout=SeqTask_evaluate(what,varargin)
% Different functions for the evaluation of
% Different neuronal networks
baseDir = '/Users/jdiedrichsen/Dropbox (Diedrichsenlab)/projects/RNN_sequence';

switch(what)
    case 'Run_all' 
        simparams=SeqTask_evaluate('Get_simparamsTest');
        [D,v_inputtrain_T,m_targettrain_T]=SeqTask_evaluate('Get_singleTrialTest',simparams);
        net = SeqTask_evaluate('Get_network','GaussianTest');
        data = SeqTask_evaluate('Run_simulation',net,v_inputtrain_T);
    case 'Get_simparamsTest'
        simparams.taskfun=@SeqTask_taskfun_gaussian2; 
        simparams.numEpisodes=20; 
        simparams.numTargets=5; 
        simparams.numTrials=1;
        simparams.memRange=[80 0];
        simparams.preTime=10;
        simparams.moveTime=140; 
        simparams.noGoFreq=0; 
        varargout={simparams}; 
    case 'Get_singleTrialTest' 
        simparams=varargin{1}; 
        D.targs=perms([1:5]);
        N=size(D.targs,1); 
        D.episode = [1:N]'; 
        D.trial = ones(N,1);  
        D.noGo = zeros(N,1); 
        D.memLength = ones(N,1)*simparams.memRange(1); 
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
            listing = dir(fullfile(loadDir,sprintf('hfopt_%s_d_*.mat',name,networkum)));
            thisFile = listing(loadInd).name;
        end;
        load(fullfile(loadDir,thisFile),'net','simparams')
        disp(['Loaded: ' thisFile])
        varargout={net,simparams}; 
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
            plot(t, data{tn}{3}(j,:),cmap{j}); 
            hold on; 
        end; 
        hold off; 
        set(gca,'XLim',[1 max(t)]); 
    case 'RDM_dynamics' 
        % Generate a time resolved RDM 
        D=varargin{1}; 
        data = varargin{2}; 
        times = [5:10:250]; 
        K = size(D.episode,1); 
        H = eye(K)-ones(K)/K; 
        C=indicatorMatrix('allpairs',[1:K]); 
        for i=1:length(data) 
            Z(:,:,i) = data{i}{1};
        end; 
        figure(1); 
        output = data{1}{3}; 
        t=[1:size(output,2)]; 
        plot(t,output); 
        drawline(times); 
        figure(2); 
        for i=1:length(times) 
            Zslice=squeeze(Z(:,times(i),:))'; 
            Diff = C*Zslice; 
            RDM=squareform(sum(Diff.*Diff,2)); 
            subplot(5,5,i); 
            imagesc(sqrt(RDM),[0 3]); 
            drawline([24.5:24:100],'dir','vert'); 
            drawline([24.5:24:100],'dir','horz'); 
        end; 
    case 'RDM_predictions'
        D=varargin{1};         
        K = size(D.episode,1); 
        C=indicatorMatrix('allpairs',[1:K]); 
        RDM=[];
        for i=1:5 
            Z=indicatorMatrix('identity',D.targs(:,i)); 
            Diff = C*Z; 
            RDM(:,:,i)=squareform(sum(Diff.*Diff,2)); 
            subplot(2,5,i); 
            imagesc(sqrt(RDM(:,:,i))); 
            drawline([24.5:24:100],'dir','vert'); 
            drawline([24.5:24:100],'dir','horz'); 
        end; 
    case 'State_Space_Plot'
        % set(gcf, 'Position', [998 565 600 420], 'PaperPositionMode', 'Auto')
        % build trial-wise cmap
        D=varargin{1}; 
        data = varargin{2}; 
        
        K= size(D.episode,1); 
        T= size(data{1}{1},2); 
        dimensions = 3; 
        timeRange = [1:T-20];

        % Color maps for fingers 
        cmap= [1 0 0;0.7 0 0.7;0 0 1;0 0.7 0.7;0 0.7 0]; 
        
        % Set the time symbols 
        stampTime = [1 D.cue(1,:) D.cueend(1) D.gocue(1) max(timeRange)]; 
        stampSymbol = ['^','+','+','+','+','+','o','o','o']; 
        stampColor = [0 0 0;cmap;0.7 0.7 0.7;0 1 0;0 0 0]; 
        stampName = {'start','D1','D2','D3','D4','D5','cueend','Go','end'};
        
        % Adjust the timing symbols to current time window
        stampTime(1) = timeRange(1); 
        stampTime(end) = timeRange(end);
        stampTime = stampTime-timeRange(1)+1; 
        
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
            Z(:,:,i) = data{i}{1};
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
                    h(i)=plot3(pc(1,stampTime(i),cond), pc(2,stampTime(i),cond), pc(3,stampTime(i),cond), stampSymbol{i}, 'Color', stampColor(i,:), 'MarkerSize', 5);
                end; 
            elseif dimensions == 2
                % Todo 
            end
        end        
        legend(h,stampName); 
        hold off; 
end