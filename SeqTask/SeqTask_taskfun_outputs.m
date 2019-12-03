function [m_targettrain_T,D] = SeqTask_taskfun_outputs(D,simparams)
% Generate inpt time courses for an epsiode of trials
numtrials = length(D.trial);
targShape = gausswin(25,3);
targShape = targShape / max(targShape) * 0.6;
m_targettrain_T={};
trialstart = 1; % Time index for first trial
m_targettrain_T=zeros(5,D.end(end));
for i = 1:numtrials% repetitions
    tStart = D.gocue(i,1)+9;
    numTargets =sum(~isnan(D.targs(i,:))); 
    for j = 1:numTargets 
        m_targettrain_T(D.targs(i,j), tStart : tStart + length(targShape) - 1) = targShape;
        tStart = tStart + length(targShape);
    end
end
