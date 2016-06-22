
actions = {
    'cup - drink'
    'cup - pound'
    'cup - shake'
    'cup - move around'
    'cup - pour'
    'stone - pound'
    'stone - move around'
    'stone - play'
    'stone - grind'
    'stone - carve'
    'sponge - squeeze'
    'sponge - flip'
    'sponge - wash'
    'sponge - wipe'
    'sponge - scratch'
    'spoon - scoop'
    'spoon - stir'
    'spoon - hit'
    'spoon - eat'
    'spoon - sprinkle'
    'knife - cut'
    'knife - chop'
    'knife - poke a hole'
    'knife - peel'
    'knife - spread'
};

load('eval_result.mat', 'gt_all', 'pred_all');

M = length(actions);

cm_count = confusionmat(gt_all,pred_all,'order',[1:M]);
cm_mat = cm_count./repmat(sum(cm_count,2),[1, size(cm_count,2)]);
cm_mat(isnan(cm_mat))=0;
% draw_confusion_matrix(cm_mat);
draw_confusion_matrix(cm_mat, actions);
ylabel('Ground truth action classes', 'FontSize', 14);
xlabel('Predicted action classes', 'FontSize', 14);
% colormap jet
colorbar