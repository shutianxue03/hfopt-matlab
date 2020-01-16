function [v_inputtrain_T, m_targettrain_T, did_change_train_data, net, all_simdata] = ...
    ContTask_taskfun(net, v_inputtrain_T, m_targettrain_T, simparams, all_simdata, do_inputs, do_targets)
% Generates inputs and / or target shapes for design specified in simparams
did_change_train_data = false;
j=1;

% Produce the targets, inputs and instruction inputs
if do_inputs
    net.taskData=[];
    for i = 1:simparams.numEpisodes   % Number of episodes
        D=[];
        mem = SeqTask_getValue(simparams.memorySpan);  % Memory span for this episode 
        for t = 1:simparams.numTargets % Number of trials in episode
            D.episode(t,1)=i;
            D.targetNum(t,1)=t;
            D.memorySpan(t,1)=mem;
            % Determine targets
            D.target(t,1) = simparams.targetset(randsample(length(simparams.targetset),1,1));
            % Memory interval length
        end
        net.taskData = addstruct(net.taskData,D);
    end
    [v_inputtrain_T,net.taskData] = ContTask_taskfun_inputs(net.taskData,simparams);
    did_change_train_data = true;
end

% Produce the desired outout
if do_targets
    m_targettrain_T = ContTask_taskfun_outputs(net.taskData,simparams);
end; 