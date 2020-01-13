function [v_inputtrain_T,D] = ContTask_taskfun_inputs(D,simparams)
% Generate inpt time courses for an epsiode of trials
numtrials = length(D.trial);
v_inputtrain_T={};
ep = unique(D.episode)';
for e=ep
    inp = [];
    start = 1; % Time index for first trial
    tn = find(D.episode==e);
    for i = 1:length(tn) % trials within the
        % Input time course
        numTargets = sum(~isnan(D.targs(tn(i),:)));
        D.start(tn(i),1) = start;
        t=start+simparams.preTime;
        for j = 1:numTargets
            inp(D.targs(tn(i),j), t: t+simparams.cueOn-1) = 1;
            D.cue(tn(i),j)=t;
            t=t+simparams.cueOn + simparams.cueOff; 
        end
        D.cueend(tn(i),1) = t; 
        
        if ~D.noGo(tn(i))         % Go trial: 
            D.gocue(tn(i),1) = D.cueend(tn(i))+D.memLength(tn(i))-1;
            inp(6,D.gocue(tn(i),1) : D.gocue(tn(i),1)+9) = 1;
            D.end(tn(i),1) = D.gocue(tn(i)) + simparams.RT + simparams.moveTime;
        else                      % Nogo trial 
            D.gocue(tn(i),1) = NaN; 
            D.end(tn(i),1) = D.cueend(tn(i))+D.memLength(tn(i))+ simparams.moveTime; % IPI for nogo-trial 
        end
        start = D.end(tn(i),1)+1; % Start of next trial
    end
    inp(6,D.end(tn(i),1))=0; 
    v_inputtrain_T{e}=inp;
end
