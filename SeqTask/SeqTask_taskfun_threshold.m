function [v_inputtrain_T, m_targettrain_T, did_change_train_data, net, all_simdata] = ...
    SeqTask_taskfun_threshold(net, v_inputtrain_T, m_targettrain_T, simparams, all_simdata, do_inputs, do_targets)

did_change_train_data = false;
if do_inputs
    rng(1)
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
        end
    end
    did_change_train_data = true;
end

if do_targets
    rng(1)
    for i = 1:length(v_inputtrain_T)
        try
            out = simparams.forwardPass{i}{3};
        catch
            out = zeros(5, size(v_inputtrain_T{i},2));
        end
        m_targettrain_T{i} = out;
        trialStart = 1;
        for tr = 1:simparams.batchSize
            noGo = false;
            if rand < simparams.noGoFreq
                noGo = true;
            end
            goalTargs = randsample(5,simparams.numTargets,false);     
            cueLength = simparams.preTime + simparams.numTargets*10;
            memLength = simparams.memRange(1);
            if simparams.memRange(2) > 0
                memLength = memLength + randsample(simparams.memRange(2),1);
            end

            m_targettrain_T{i}(:,trialStart:trialStart+cueLength+memLength) = 0;
            
            targCount = 1;
            thisTarg = goalTargs(targCount);
            pressedTargs = []; penaltyTargs = [];
            pressed = false;
            for t = trialStart+cueLength+memLength+9 : trialStart+cueLength+memLength+simparams.moveTime-1
                for targ = 1:5
                    if targ == thisTarg
                        if ~pressed
                            if out(targ,t) < 0.5
                                m_targettrain_T{i}(targ,t) = 1;
                            else
                                pressedTargs(end+1) = targ;
                                pressed = true;
                                pressedCount = 5;
                            end
                        else
                            pressedCount = pressedCount - 1;
                            if pressedCount == 0
                                targCount = targCount + 1;
                                pressed = false;
                                pressedCount = 5;
                                penaltyTargs(end+1) = targ;
                            end
                        end
                    else
                        if out(targ,t) > 0.5 && ~ismember(targ, pressedTargs)
                            m_targettrain_T{i}(targ,t) = 0;
                        end
                        if ~ismember(targ, goalTargs) || ismember(targ, penaltyTargs)
                            %m_targettrain_T{i}(targ,t) = m_targettrain_T{i}(targ,t) * 0.95;
                        end
                    end
                end
                if targCount <= length(goalTargs)
                    thisTarg = goalTargs(targCount);
                else
                    break;
                end
            end
            if noGo
                m_targettrain_T{i}(:,trialStart : trialStart+cueLength+memLength+simparams.moveTime-1) = 0;
            end
            trialStart = trialStart+cueLength+memLength+simparams.moveTime;
        end
        %m_targettrain_T{i}(m_targettrain_T{i} ~= 1) = m_targettrain_T{i}(m_targettrain_T{i} ~= 1) * 0.99;
        %m_targettrain_T{i}(m_targettrain_T{i} < 0) = 0;
    end
end




end