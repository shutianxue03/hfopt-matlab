function [m_targettrain_T,D] = SeqTask_taskfun_outputs(D,simparams)
% Generate inpt time courses for an epsiode of trials
numtrials = length(D.trial);
targShape = gausswin(simparams.forceWidth,3);
targShape = targShape / max(targShape) * 0.6;
m_targettrain_T={};
ep = unique(D.episode)';
for e=ep
    inp = [];
    tn = find(D.episode==e);
    trialstart = 1; % Time index for first trial
    m_targettrain_T{e}=zeros(5,D.end(tn(end))); % Preallocate 
    for i = 1:length(tn) % trials within the
        if (~D.noGo(tn(i)))
            tStart = D.gocue(tn(i),1)+simparams.RT;
            numTargets =sum(~isnan(D.targs(tn(i),:))); 
            for j = 1:numTargets
                m_targettrain_T{e}(D.targs(tn(i),j), tStart : tStart + length(targShape) - 1) = targShape;
                tStart = tStart + simparams.forceIPI;
            end 
        end
    end
    m_targettrain_T{e}(1,D.end(tn(end)))=0; 
end