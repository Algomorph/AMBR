
% subjects = {'gui', 'kos', 'fer', 'and', 'mic'};
subjects = {'and', 'fer', 'gui', 'kos', 'mic'};
subject_ids = [4, 3, 1, 2, 5];
% subjects = {'gui'};
objects = {'cup', 'stone', 'sponge', 'spoon', 'knife', 'spatula'};
% objects = {'cup'};
data_root = './Tel2015Data0704';

for oi = 1 : length(objects)
    
    label_file = sprintf('%s/frontal_label/action_label_%s.txt', data_root, objects{oi});
    dataset_dir = sprintf('%s/datasets/', data_root);
    if ~exist(dataset_dir, 'dir')
        mkdir(dataset_dir);
    end
    
    feat_file = sprintf('%s/%s_feat.mat', dataset_dir, objects{oi});
    json_file = sprintf('%s/%s_db.json', dataset_dir, objects{oi});
    
    fin = fopen(label_file, 'r');
    if fin == -1,
        fprintf('Cannot open label file.\n');
        return,
    end
    fout = fopen(json_file, 'w');
    if fout == -1,
        fprintf('Cannot open json file for writing.\n');
        return,
    end
    
    % readin feature files
    sub_feats = {};
    for i = 1 : length(subjects)
        si = subject_ids(i);
        sub_feats{si} = load(sprintf('%s/frontal_feats/%s_%s_feats.mat', data_root, subjects{i}, objects{oi}), 'feats', 'id_list');
    end
    
    feat_all = cell(1, 150);
    segment_id = 0;
    offset = 0;
    A_prev = [];
    
    fprintf(fout, '[');
    while 1,
        strline = fgetl(fin);
        % fprintf('%s', strline);
        if ~ischar(strline),
            break;
        end
        if strcmp(strline, '') == 1,
            continue;
        end
        if strline(1) == '#', 
            continue;
        end
        
        A = sscanf(strline, '%d %d %d %d %d %d');
        if (length(A) < 6),
            fprintf('data format error: "%s"\n', strline);
            error;
        end
        % cut sequence tails
        new_end = A(4) - 30;
        new_end = max(new_end, A(3) + 30);
        if (new_end > A(4))
            keyboard;
        end
        A(4) = new_end;
        
        
        if (segment_id > 0)
            % add negative samples

%             if (A(1) == A_prev(1))
%                 A0 = [A(1) A_prev(4)+31 A_prev(4)+31 A(2)-1 0 0];
%                 % parsing
%                 i = A0(1);
%                 id_list = sub_feats{i}.id_list;
%                 start_fid = find(id_list == A0(2), 1);
%                 touch_fid = find(id_list == A0(3), 1);
%                 end_fid = find(id_list == A0(4), 1);
%                 seq_len = A0(4)-A0(2)+1;
%                 preseq_len = A0(3)-A0(2)+1;
%                 if (seq_len > 15) % remove too short sequence
%                     fprintf(fout, ',\n{');
%                     fprintf(fout, '"sub_id": %d, ', A0(1));
%                     fprintf(fout, '"subject": "%s", ', subjects{find(subject_ids==i,1)});
%                     fprintf(fout, '"object": "%s", ', objects{oi});
%                     fprintf(fout, '"segment_id": %d, ', segment_id);
%                     fprintf(fout, '"start_frame": %d, ', A0(2));
%                     fprintf(fout, '"touch_frame": %d, ', A0(3));
%                     fprintf(fout, '"end_frame": %d, ', A0(4));
%                     fprintf(fout, '"start_fid": %d, ', offset);
%                     fprintf(fout, '"touch_fid": %d, ', offset+A0(3)-A0(2));
%                     fprintf(fout, '"end_fid": %d, ', offset+A0(4)-A0(2));
%                     fprintf(fout, '"seq_len": %d, ', seq_len);
%                     fprintf(fout, '"preseq_len": %d, ', preseq_len);
%                     fprintf(fout, '"attention_type": %d, ', A0(5));
%                     fprintf(fout, '"test_flag": %d', A0(6));
%                     fprintf(fout, '}');
%                     
%                     segment_id = segment_id + 1;
%                     feat_all{segment_id} = sub_feats{i}.feats(:, start_fid:end_fid);
%                     offset = offset + seq_len;
%                     
%                 end
%             end

            fprintf(fout, ',\n');
        end
        
        
        % parsing
        i = A(1);
        id_list = sub_feats{i}.id_list;
        start_fid = find(id_list == A(2), 1);
        touch_fid = find(id_list == A(3), 1);
        end_fid = find(id_list == A(4), 1);
        seq_len = A(4)-A(2)+1;
        preseq_len = A(3)-A(2)+1;
        
        fprintf(fout, '{');
        fprintf(fout, '"sub_id": %d, ', A(1));
        fprintf(fout, '"subject": "%s", ', subjects{find(subject_ids==i,1)});
        fprintf(fout, '"object": "%s", ', objects{oi});
        fprintf(fout, '"segment_id": %d, ', segment_id);
        fprintf(fout, '"start_frame": %d, ', A(2));
        fprintf(fout, '"touch_frame": %d, ', A(3));
        fprintf(fout, '"end_frame": %d, ', A(4));
        fprintf(fout, '"start_fid": %d, ', offset);
        fprintf(fout, '"touch_fid": %d, ', offset+A(3)-A(2));
        fprintf(fout, '"end_fid": %d, ', offset+A(4)-A(2));
        fprintf(fout, '"seq_len": %d, ', seq_len);
        fprintf(fout, '"preseq_len": %d, ', preseq_len);
        fprintf(fout, '"attention_type": %d, ', A(5));
        fprintf(fout, '"test_flag": %d', A(6));
        fprintf(fout, '}');
        
        segment_id = segment_id + 1;
        feat_all{segment_id} = sub_feats{i}.feats(:, start_fid:end_fid);
        offset = offset + seq_len;
        
        assert( end_fid - start_fid + 1 == seq_len );        
        A_prev = A;

    end
    
    fprintf(fout, ']');
    
    fclose(fin);
    fclose(fout);
    
    feat_all = feat_all(1:segment_id);
    
    feats = cell2mat(feat_all);
%     save(feat_file, 'feats');
    
    fprintf('parsed %d sequences.\n', segment_id);
    
    
end


