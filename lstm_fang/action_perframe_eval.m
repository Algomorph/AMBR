% close all;

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

object_list = {'cup', 'stone', 'sponge', 'spoon', 'knife'};

% subject_list = {'and', 'mic', 'fer', 'kos', 'gui'};
subject_list = {'and', 'fer', 'gui', 'kos', 'mic'};
subject_list = {'gui'};

subject_ids = [1];

topn = 5;

fullimage_dir = 'Tel2015Data0704/frontal_rgb/';
patch_dir = 'Tel2015Data0704/frontal_patch/';
label_dir = 'Tel2015Data0704/frontal_label/';

output_dir = 'ResultsAnalysis/supp';
if ~exist(output_dir, 'dir')
  mkdir(output_dir);
end


label_all = [];
for oi = 1 : length(object_list)
  label = load([label_dir, 'action_label_' object_list{oi} '.txt']);
  label_all = [label_all; [label, ones(size(label,1),1)*oi]];
end

gt_all = [];
pred_all = [];

precision = [];
recall = [];

for si = 1 : length(subject_list)
  subject = subject_list{si};
%   label_test = label_all(label_all(:,1)==subject_ids(si), :);
  
  model_file = sprintf('test_results8_default_%s/lstm_default_%s_model_result_all.mat', subject, subject);

  data = load(model_file);  
  select_idx = [1:5, 26:30, 51:55, 76:80, 101:105];
%   select_idx = [ 2    28    53     102   104 ];   % and
%   select_idx = [ 1     4     5    27    51    55   104   105 ];   % fer
  
  gts = data.gts(select_idx) + 1;
  preds_all = data.preds_all(select_idx);
  start_frame = data.start_frame(select_idx);
  end_frame = data.end_frame(select_idx);
  objs_tmp = mat2cell(data.object(select_idx,:), ones(1, length(select_idx)), [size(data.object,2)]);
  objs = cellfun(@(x) find(ismember(object_list, strtrim(x)),1), objs_tmp);
  
  
  figure(3);
  for i = 1 : length(preds_all)
    subplot(5,5,i);
    imagesc(squeeze(preds_all{i}));
  end
  
  
  for i = 1 : length(preds_all)
    
    figure(1); 
    imagesc(squeeze(preds_all{i})); colorbar;
    
    [scores, orders] = sort(squeeze(preds_all{i}), 2, 'descend');
    assert(end_frame(i) - start_frame(i) + 1 == size(scores, 1));
    
    oi = objs(i);
    fid = 0; 
    for j = start_frame(i) : 2 : end_frame(i)
      im = imread([fullimage_dir subject '_' object_list{oi} '_rgb', sprintf('/%08d.jpg', j)]);
      subdir = sprintf('%s/%s_rst%d', output_dir, subject, i);
      if ~exist(subdir, 'dir'), mkdir(subdir); end
      outfile = sprintf('%s/%06d.jpg', subdir, fid);
      fid = fid + 1;

      ttim = im;
      for ti = 1:topn
        ind = j-start_frame(i)+1;
%         fprintf(' %d - %s\n', ti, actions{orders(j-label_test(i,2)+1, ti)});

        if (orders(ind, ti) == gts(i))
          ttim = insertText(ttim, [0.01, ti*25-20], ...
          sprintf('%s: %.03f                        ', actions{orders(ind, ti)}, scores(ind, ti)), ...
          'TextColor', 'g', 'FontSize', 18, ...
          'BoxOpacity', 0.3);
        else
          ttim = insertText(ttim, [0.01, ti*25-20], ...
          sprintf('%s: %.03f                        ', actions{orders(ind, ti)}, scores(ind, ti)), ...
          'TextColor', 'g', 'FontSize', 18, ...
          'BoxOpacity', 0);
        end
        
      end
      
      figure(2);
      imshow(ttim);
      
      imwrite(ttim, outfile);
      pause(0.2);
    end
    
%     close all;
    pause(1);
  end

end


% cm_count = confusionmat(gt_all,pred_all,'order',[1:M]);
% cm_mat = cm_count./repmat(sum(cm_count,2),[1, size(cm_count,2)]);
% cm_mat(isnan(cm_mat))=0;
% draw_confusion_matrix(cm_mat);
% ylabel('Ground truth action classes', 'FontSize', 14);
% xlabel('Predicted action classes', 'FontSize', 14);
% colormap jet
% 
% rst_per_class = [nanmean(precision, 2), nanmean(recall, 2)] * 100.0;
% rst_all = mean(rst_per_class, 1);
% 
% 
% % print in latex format
% fprintf(' Action &  Precision  &  Recall \\\\ \n');
% fprintf(' \\hline \n');
% for i = 1 : length(actions)
%   fprintf(' %s &  %.01f\\%%  &  %.01f\\%% \\\\ \n', actions{i}, rst_per_class(i, 1), rst_per_class(i, 2));
% end
% 
% fprintf(' \\hline \n');
% fprintf(' \\hline \n');
% fprintf(' Avg. &  %.01f\\%%  &  %.01f\\%% \\\\ \n', rst_all(1), rst_all(2));
% 
% 
