
clear;

dataset = 'temp';
label_file = 'section_label_full_action.txt';

load(sprintf('./data/%s/vgg_feats.mat', dataset), 'feats', 'id_list');
json_file = sprintf('./data/%s/sequences.json', dataset);

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

seq_id = 0;
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
    
    A = sscanf(strline, '%d %d %d %d %d');
    if (length(A) < 5),
        fprintf('data format error: "%s"\n', strline);
        error;
    end
    
    if (seq_id > 0)
        fprintf(fout, ',\n');
    end
    
    fprintf(fout, '{');
    fprintf(fout, '"sub_id": %d, ', A(1));
    fprintf(fout, '"seq_id": %d, ', A(2));
    fprintf(fout, '"start_frame": %d, ', A(3));
    fprintf(fout, '"end_frame": %d, ', A(4));
    fprintf(fout, '"start_fid": %d, ', find(id_list == A(3), 1));
    fprintf(fout, '"end_fid": %d, ', find(id_list == A(4), 1));
    fprintf(fout, '"seq_len": %d, ', A(4)-A(3)+1);
    fprintf(fout, '"attention_type": %d', A(5));
    fprintf(fout, '}');
    
    seq_id = seq_id + 1;
end
fprintf(fout, ']');

fclose(fin);
fclose(fout);

fprintf('parsed %d sequences.\n', seq_id);

