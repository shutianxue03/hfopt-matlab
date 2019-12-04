function [v_inputtrain_T,D] = SeqTask_taskfun_inputs(D,simparams)
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
        cueLength = simparams.preTime + numTargets*10;
        lengthTrial = cueLength+D.memLength(tn(i))+10+simparams.moveTime;
        tCourse = zeros(6,lengthTrial);
        if ~D.noGo(i)
            tCourse(6,cueLength+D.memLength(tn(i)) : cueLength+D.memLength(tn(i))+9) = 1;
        end
        for j = 1:numTargets
            tCourse(D.targs(tn(i),j),simparams.preTime + (j-1)*10 + 1 : simparams.preTime + j*10) = 1;
        end
        inp = cat(2, inp, tCourse);
        D.start(tn(i),1) = start;
        D.end(tn(i),1) = start + lengthTrial-1;
        D.cueend(tn(i),1) = start + cueLength-1; 
        D.gocue(tn(i),1) = start+cueLength+D.memLength(i)-1;
        start = D.end(tn(i),1)+1; % Start of next trial
    end;
    v_inputtrain_T{e}=inp;
end
