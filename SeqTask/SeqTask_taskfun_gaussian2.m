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
    for i = 1:simparams.numTrials   % trials 
        v_inputtrain_T{i} = [];
        for tr = 1:simparams.batchSize % repetitions 
            % Go or NoGo ? 
            noGo = false;
            if rand < simparams.noGoFreq
                noGo = true;
            end
            % Determine targets 
            targs = randsample(5,simparams.numTargets,simparams.withreplace);
            cueLength = simparams.preTime + simparams.numTargets*10;
            % Memory interval length 
            memLength = simparams.memRange(1);
            if simparams.memRange(2) > 0
                memLength = memLength + randsample(simparams.memRange(2),1);
            end
            
            % Input time course
            lengthTrial = cueLength+memLength+simparams.moveTime; 
            tCourse = zeros(6,);
            if ~noGo
                tCourse(6,cueLength+memLength : cueLength+memLength+9) = 1;
            end
            for j = 1:length(targs)
                tCourse(targs(j),simparams.preTime + (j-1)*10 + 1 : simparams.preTime + j*10) = 1;
            end
            v_inputtrain_T{i} = cat(2, v_inputtrain_T{i}, tCourse);
            
            net.taskData.trial(j,1) = i;
            net.taskData.rep(j,1) = tr;
            net.taskData.start(j,1) = 
            net.taskData.end(j,1) = 
            net.taskData.gocue(j,1) = net.taskData.start(j,1)+cueLength+memLength;
            
            
            net.taskData.noGo(j,1) = noGo;
            net.taskData.targs(j,:) = targs;
            net.taskData.memLength(j,1) = memLength;
        end
    end
    did_change_train_data = true;
end

% Produce the desired outout
if do_targets
    targShape = gausswin(25,3);
    targShape = targShape / max(targShape) * 0.6;
    for i = 1:length(v_inputtrain_T)
        m_targettrain_T{i} = zeros(5, size(v_inputtrain_T{i},2));
        trialStart = 1;
        for tr = 1:simparams.batchSize
            noGo = net.taskData{i,tr}.noGo;
            goalTargs = net.taskData{i,tr}.targs;
            memLength = net.taskData{i,tr}.memLength;  
            cueLength = simparams.preTime + simparams.numTargets*10;
            if ~noGo
                tStart = trialStart + cueLength + memLength + 9;
                for j = 1:length(goalTargs)
                    m_targettrain_T{i}(goalTargs(j), tStart : tStart + length(targShape) - 1) = targShape;
                    tStart = tStart + 10;
                end  
            end
            trialStart = trialStart+cueLength+memLength+simparams.moveTime;
        end
    end
end

end