function [m_targettrain_T,D] = ContTask_taskfun_outputs(D,simparams)
% Generate inpt time courses for an epsiode of trials
targShape = gausswin(simparams.forceWidth,3)';
targShape = targShape / max(targShape) * 0.6;
m_targettrain_T={};
ep = unique(D.episode)';
for e=ep
    inp = [];
    tn = find(D.episode==e);
    trialstart = 1; % Time index for first trial
    m_targettrain_T{e}=zeros(5,D.end(tn(end))); % Preallocate 
    for i = 1:length(tn) % trials within the
        t = D.goOnset(tn(i),1)+simparams.RT;
        m_targettrain_T{e}(D.target(tn(i),1), t: t+simparams.forceWidth - 1) = targShape;
    end 
    m_targettrain_T{e}(1,D.end(tn(end)))=0; 
end