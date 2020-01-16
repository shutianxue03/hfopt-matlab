function [v_inputtrain_T,D] = ContTask_taskfun_inputs(D,simparams)
% Generate inpt time courses for an epsiode of trials
v_inputtrain_T={};
ep = unique(D.episode)';
for e=ep
    inp = [];
    tn = find(D.episode==e);
    numTargets = length(tn);
    cueTarget = 0; % Targets current in memory
    goTarget = 0; % Targets already executed
    memory = nan(1,max(D.memorySpan));
    t = SeqTask_getValue(simparams.preTime);  % Time index for first cue
    while goTarget<numTargets
        % Check if you want to present a new cue
        if (cueTarget<numTargets)
            cueTarget=cueTarget+1;
            i=tn(cueTarget);        % target index that is cued
            D.cueOnset(i,1)=t;
            inp(D.target(i), t:t+simparams.cueDur-1) = 1;
            t=t+simparams.cueDur;
            % Now put the new cue in short-term memory
            emptyIndx=find(isnan(memory));
            memory(emptyIndx(1))=D.target(i);
            D.memoryCue(i,:)=memory;
            t=t+SeqTask_getValue(simparams.postCue);
        else
            t=t+SeqTask_getValue(simparams.postCue);
        end;
        if ((cueTarget-goTarget == D.memorySpan(tn(1))) || ...
                (cueTarget == numTargets))
            goTarget=goTarget+1;   % target number that is executed
            i=tn(goTarget);
            D.goOnset(i,1)=t;
            D.memoryGo(i,:)=memory;
            memory = [memory(2:end) NaN]; 
            inp(6,t:t+simparams.goDur-1)=1;
            D.peakPress(i,1) = t+ simparams.RT + round(simparams.forceWidth/2);
            D.endPress(i,1) = t+ simparams.RT + simparams.forceWidth;
            t=D.endPress(i)+SeqTask_getValue(simparams.postPress);
            D.end(i,1)=t; 
        else
            t=t+simparams.RT + SeqTask_getValue(simparams.postPress);
        end;
    end
    inp(6,D.end(tn(end),1))=0;
    v_inputtrain_T{e}=inp;
end