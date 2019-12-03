function [v_inputtrain_T, m_targettrain_T, did_change_train_data, net, all_simdata] = ...
    SeqTask_taskfun_gaussian2(net, v_inputtrain_T, m_targettrain_T, simparams, all_simdata, do_inputs, do_targets)
% Generates inputs and / or target shapes for design specified in simparams
did_change_train_data = false;
j=1; 
if ~isfield(simparams,'withreplace')
    simparams.withreplace=true; 
end; 
% Produce the targets, inputs and instruction inputs
if do_inputs
    net.taskData=[]; 
    for i = 1:simparams.numEpisodes   % trials 
        D=[]; 
        for tr = 1:simparams.batchSize % repetitions 
            D.epsiode(tr,i)=i;
            D.trial(tr,1)=tr; 
            
            % Go or NoGo ? 
            D.noGo(tr,1) = false;
            if rand < simparams.noGoFreq
                D.noGo(tr,1) = true;
            end
            % Determine targets 
            D.targs(tr,:) = randsample(5,simparams.numTargets,simparams.withreplace);
            % Memory interval length 
            D.memLength(tr,1) = simparams.memRange(1);
            if simparams.memRange(2) > 0
                D.memLength(tr,1)  = D.memLength(tr,1) + randsample(simparams.memRange(2),1);
            end
        end; 
        v_inputtrain_T{i} = SeqTask_taskfun_inputs(D,simparams); 
        net.taskData = addstruct(net.taskData,D); 
        end
    end
    did_change_train_data = true;
end

% Produce the desired outout
if do_targets
    for i = 1:length(v_inputtrain_T)
        m_targettrain_T{i} = SeqTask_taskfun_outputs(net.taskData,simparams); 
    end
end; 