function [v_inputtrain_T, m_targettrain_T, did_change_train_data, net, all_simdata] = ...
    SeqTask_taskfun_gaussian(net, v_inputtrain_T, m_targettrain_T, simparams, all_simdata, do_inputs, do_targets)
% Generates inputs and / or target shapes for design specified in simparams
did_change_train_data = false;
j=1;

% Produce the targets, inputs and instruction inputs
if do_inputs
    net.taskData=[];
    for i = 1:simparams.numEpisodes   % Number of episodes
        D=[];
        for tr = 1:simparams.numTrials % Number of trials in episode
            D.episode(tr,1)=i;
            D.trial(tr,1)=tr;
            
            % Go or NoGo ?
            D.noGo(tr,1) = false;
            if rand < simparams.noGoFreq
                D.noGo(tr,1) = true;
            end
            
            D.numTargets(tr,:)=simparams.numTargets(unidrnd(length(simparams.numTargets)));
            % Determine targets
            D.targs(tr,1:5)=NaN; 
            D.targs(tr,1:D.numTargets(tr,:)) = simparams.targetset(randsample(length(simparams.targetset),D.numTargets(tr,:),simparams.withreplace));
            % Memory interval length
            if (length(simparams.memPeriod)==1) 
                D.memLength(tr,1) = simparams.memPeriod(1);
            else 
                D.memLength(tr,1)  = simparams.memPeriod(1) + ...
                    randsample(simparams.memPeriod(2)-simparams.memPeriod(1),1);
            end
        end
        net.taskData = addstruct(net.taskData,D);
    end
    [v_inputtrain_T,net.taskData] = SeqTask_taskfun_inputs(net.taskData,simparams);
    did_change_train_data = true;
end

% Produce the desired outout
if do_targets
    m_targettrain_T = SeqTask_taskfun_outputs(net.taskData,simparams);
end; 