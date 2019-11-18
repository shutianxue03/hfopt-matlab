function [v_inputtrain_T, m_targettrain_T, did_change_train_data, net, all_simdata] = ...
    SeqTask_taskfun_gaussian(net, v_inputtrain_T, m_targettrain_T, simparams, all_simdata, do_inputs, do_targets)

did_change_train_data = false;
if do_inputs
    for i = 1:simparams.numTrials   
        v_inputtrain_T{i} = [];
        for tr = 1:simparams.batchSize
            noGo = false;
            if rand < simparams.noGoFreq
                noGo = true;
            end
            targs = randsample(5,simparams.numTargets,false);
            cueLength = simparams.preTime + simparams.numTargets*10;
            memLength = simparams.memRange(1);
            if simparams.memRange(2) > 0
                memLength = memLength + randsample(simparams.memRange(2),1);
            end
            tCourse = zeros(6,cueLength+memLength+simparams.moveTime);
            if ~noGo
                tCourse(6,cueLength+memLength : cueLength+memLength+9) = 1;
            end
            for j = 1:length(targs)
                tCourse(targs(j),simparams.preTime + (j-1)*10 + 1 : simparams.preTime + j*10) = 1;
            end
            v_inputtrain_T{i} = cat(2, v_inputtrain_T{i}, tCourse);
            net.taskData{i,tr}.noGo = noGo;
            net.taskData{i,tr}.targs = targs;
            net.taskData{i,tr}.memLength = memLength;
        end
    end
    did_change_train_data = true;
end

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