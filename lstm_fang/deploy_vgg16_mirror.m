
caffe_root = '/home/fwang/Programs/caffe';
addpath(genpath(sprintf('%s/matlab/', caffe_root)));
model_def_file = sprintf('./vgg_model/vgg16_deploy.prototxt');
model_file = sprintf('./vgg_model/VGG_ILSVRC_16_layers.caffemodel');
use_gpu = true;
input_dim = 224;
batch_size = 50;
feat_dim = 4096;
mean_vec = single([103.939, 116.779, 123.68]);
mean_img = repmat(reshape(mean_vec, [1,1,3]), [input_dim, input_dim, 1]);

matcaffe_init(use_gpu, model_def_file, model_file);

subjects = {'yezhou', 'cornellia'};
objects = {'sponge', 'cup'};
data_root = './YiData/frames';
out_dir = sprintf('%s/feats', data_root);
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

for si = 1 : length(subjects)
    
    out_file = sprintf('%s/%s_feats.mat', out_dir, subjects{si});
    bbox_file = sprintf('%s/mirror_%s_bbox.txt', data_root, subjects{si});
    patch_dir = sprintf('%s/mirror_%s_patch', data_root, subjects{si});
    
    %         if exist(out_file, 'file'),
    %             continue;
    %         end
    
    f = fopen(bbox_file, 'r');
    if (f == -1)
        fprintf('Error: cannot open file %s.\n', bbox_file);
    end
    
    % parse the frame id list
    id_list = [];
    ct = 0;
    while 1,
        strline = fgetl(f);
        if ~ischar(strline)
            break;
        end
        if strcmp(strline, '') == 1,
            continue;
        end
        
        A = sscanf(strline, '%d %d %d %d %d');
        if (length(A) < 5),
            error,
        end
        imid = A(1);
        ct = ct + 1;
        id_list(ct) = imid;
    end
    
    fprintf('Processing patch folder: %s ... \n', patch_dir);
    tic;
    
    % readin all patches
    ori_im_num = length(id_list);
    im_num = ori_im_num;
    % padding to match batch_size
    if mod(ori_im_num, batch_size) > 0
        im_num = ceil(im_num / batch_size) * batch_size;
        for i = ori_im_num + 1 : im_num
            ct = ct + 1;
            id_list(ct) = imid;
        end
    end
    
    
    fprintf('    reading patches ... \n');
    input_data = zeros(input_dim, input_dim, 3, im_num, 'single');
    parfor i = 1 : ori_im_num
        im = imread(sprintf('%s/%08d.png', patch_dir, id_list(i)));
        input_data(:,:,:,i) = single(im) - mean_img;
    end
    % padding features
    if ori_im_num < im_num,
        input_data(:,:,:,ori_im_num + 1 : im_num) = repmat(input_data(:,:,:,ori_im_num), [1, 1, 1, im_num-ori_im_num]);
    end
    
    batch_num = ceil(im_num / batch_size);
    input_batches = squeeze(mat2cell(input_data, [input_dim], [input_dim], [3], repmat(batch_size, [1, batch_num])));
    
    
    % do forward pass to get scores
    % scores are now Width x Height x Channels x Num
    fprintf('    calculating features ... \n');
    feats = zeros(feat_dim, im_num, 'single');
    for i = 1 : batch_num
        batch_feat = caffe('forward', input_batches(i));
        feats(:, (i-1)*batch_size+1:i*batch_size) = squeeze(batch_feat{1});
    end
    
    fprintf('    saving ... \n');
    save(out_file, 'feats', 'id_list');
    toc;
    
end
