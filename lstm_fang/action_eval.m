
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
%     'spatula - flip'
%     'spatula - lift'
%     'spatula - cut'
%     'spatula - flatten'
%     'spatula - scratch'
};

subject_list = {'and', 'mic', 'fer', 'kos', 'gui'};

gt_all = [];
pred_all = [];

precision = [];
recall = [];

for si = 1 : length(subject_list)
  subject = subject_list{si};

  model_file = sprintf('test_results8_default_%s/lstm_default_%s_model_result.mat', subject, subject);

  load(model_file, 'scores', 'gts');
  M = size(scores, 2);
  [~, pred] = max(scores, [], 2);
  gt = double(gts(:)+1);
  
  cm_count_sub = confusionmat(gt, pred, 'order', [1:M]);
  tp = diag(cm_count_sub);
  cls_count = sum(cm_count_sub, 2);
  fp = sum(cm_count_sub, 1)'-tp;
  fn = cls_count-tp;
  
  prec = tp ./ (tp + fp);
  prec(tp==0) = 0;
  prec(cls_count==0) = NaN;
  rec  = tp ./ (tp + fn);
  rec(tp==0) = 0;
  rec(cls_count==0) = NaN;
  
  precision(:,si) = prec;
  recall(:,si) = rec;

  gt_all = [gt_all; gt];
  pred_all = [pred_all; pred];

end


cm_count = confusionmat(gt_all,pred_all,'order',[1:M]);
cm_mat = cm_count./repmat(sum(cm_count,2),[1, size(cm_count,2)]);
cm_mat(isnan(cm_mat))=0;
% draw_confusion_matrix(cm_mat);
draw_confusion_matrix(cm_mat, actions);
ylabel('Ground truth action classes', 'FontSize', 14);
xlabel('Predicted action classes', 'FontSize', 14);
% colormap jet

rst_per_class = [nanmean(precision, 2), nanmean(recall, 2)] * 100.0;
rst_all = mean(rst_per_class, 1);


% print in latex format
fprintf(' Action &  Precision  &  Recall \\\\ \n');
fprintf(' \\hline \n');
for i = 1 : length(actions)
  fprintf(' %s &  %.01f\\%%  &  %.01f\\%% \\\\ \n', actions{i}, rst_per_class(i, 1), rst_per_class(i, 2));
end

fprintf(' \\hline \n');
fprintf(' \\hline \n');
fprintf(' Avg. &  %.01f\\%%  &  %.01f\\%% \\\\ \n', rst_all(1), rst_all(2));


