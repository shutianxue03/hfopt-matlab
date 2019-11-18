function plot_stats = SeqTask_hfopt_Plot(net, simparams, funs, cg_found_better_solution, f3, random_trial_idxs, forward_pass_T, forward_pass_s, ...
    v_inputtrain_T, v_inputtrain_s, v_inputtest_t, ...
    m_targettrain_T, m_targettrain_s, m_targettest_t, all_optional_args, all_simdata, all_plot_stats)


cmap = [lines(5); 0 0 0];

figure(3)
clf
for i = 1:4
    subplot(3,4,i)
    if i == 1
        ylabel('Input')
    end
    hold on
    title(['Example trial ' num2str(i)])
    for j = [6 1:5]
        for tt = 1:size(v_inputtrain_T{i},2)
            if v_inputtrain_T{i}(j,tt) == 1
                plot(tt, j, '.', 'Color', cmap(j,:))
            end
        end
    end
    set(gca, 'YTick', 1:6, 'YTickLabels', {'1','2','3','4','5','Go'})
    axis([1 size(v_inputtrain_T{i},2) 0.5 6.5])
    subplot(3,4,i+4)
    if i == 1
        ylabel('Output')
    end
    hold on
    plot([1 size(forward_pass_T{i}{3},2)], [0.5 0.5], 'Color', [0.7 0.7 0.7])
    for j = 1:5
        plot(forward_pass_T{i}{3}(j,:)', 'Color', cmap(j,:))
    end
    axis([1 size(v_inputtrain_T{i},2) -0.05 0.9])
    subplot(3,4,i+8)
    if i == 1
        ylabel('Target output')
    end
    hold on
    plot([1 size(forward_pass_T{i}{3},2)], [0.5 0.5], 'Color', [0.7 0.7 0.7])
    for j = 1:5
        plot(m_targettrain_T{i}(j,:)', 'Color', cmap(j,:))
    end
    axis([1 size(v_inputtrain_T{i},2) -0.05 0.9])
    box off
end
drawnow

stats = [];
plot_stats = stats;
end