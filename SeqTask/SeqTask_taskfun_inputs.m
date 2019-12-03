function [v_inputtrain_T,D] = SeqTask_taskfun_inputs(D,simparams)
% Generate inpt time courses for an epsiode of trials
numtrials = length(D.trial);
v_inputtrain_T=[];
start = 1; % Time index for first trial
for i = 1:numtrials% repetitions
    % Input time course
    numTargets = sum(~isnan(D.targs(i,:))); 
    cueLength = simparams.preTime + numTargets*10;
    lengthTrial = cueLength+D.memLength(i)+10+simparams.moveTime;
    tCourse = zeros(6,lengthTrial);
    if ~D.noGo(i)
        tCourse(6,cueLength+D.memLength(i) : cueLength+D.memLength(i)+9) = 1;
    end
    for j = 1:numTargets
        tCourse(D.targs(i,j),simparams.preTime + (j-1)*10 + 1 : simparams.preTime + j*10) = 1;
    end
    v_inputtrain_T = cat(2, v_inputtrain_T, tCourse);
    D.start(i,1) = start;
    D.end(i,1) = start + lengthTrial-1;
    D.gocue(i,1) = start+cueLength+D.memLength(i);
    start = D.end(i,1)+1; % Start of next trial
end
