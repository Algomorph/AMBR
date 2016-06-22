
subjects = {'gui', 'kos', 'fer', 'and', 'mic'};
% subjects = {'gui'};
objects = {'cup', 'stone', 'sponge', 'spoon', 'knife', 'spatula'};
% objects = {'stone', 'sponge', 'spoon', 'knife', 'spatula'};
data_root = './Tel2015Data0704';

% skip the first frame
skip_num = 1;
patch_size = 224;
patch_half = patch_size / 2;

        
for si = 1 : length(subjects)
    for oi = 1 : length(objects)
        
        out_dir = sprintf('%s/frontal_patch/%s_%s_rgb_patch', data_root, subjects{si}, objects{oi});
%         if exist(out_dir, 'dir')
%             continue;
%         end
        
        if ~exist(out_dir, 'dir')
            mkdir(out_dir);
        end
        
        bbox_file = sprintf('%s/frontal_bbox/%s_%s_rgb_BBox.txt', data_root, subjects{si}, objects{oi});
        
        f = fopen(bbox_file, 'r');
        if (f == -1)
            fprintf('Error: cannot open file %s.\n', bbox_file);
        end
        
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
            
            im_file = sprintf('%s/frontal_rgb/%s_%s_rgb/%08d.jpg', data_root, subjects{si}, objects{oi}, imid);
            if ~exist(im_file, 'file')
                fprintf('Error: cannot read image file %s.\n', im_file);
                break;
            end
            
            im = imread(im_file);
            [height, width, c] = size(im);
            
            bbox_x1 = A(2); bbox_y1 = A(3); W = A(4); H = A(5);
            bbox_x2 = bbox_x1 + W;
            bbox_y2 = bbox_y1 + H;
            
            cx = floor((bbox_x1+bbox_x2)/2);
            cy = floor((bbox_y1+bbox_y2)/2);
            x0 = cx - patch_half + 1;
            y0 = cy - patch_half + 1;
            
            x0 = max(x0, 1);
            y0 = max(y0, 1);
            x0 = min(x0, width-patch_size+1);
            y0 = min(y0, height-patch_size+1);
            
            [imid, height, width, y0, y0+patch_size-1, x0, x0+patch_size-1]
            
%                 imshow(im);
%                 line([bbox_x1,bbox_x1,bbox_x2,bbox_x2,bbox_x1], [bbox_y1,bbox_y2,bbox_y2,bbox_y1,bbox_y1]);
%                 % keyboard;
%                 pause(0.1);
            
            patch = im(y0 : y0+patch_size-1, x0 : x0+patch_size-1, :);
            imwrite(patch, sprintf('%s/%08d.png', out_dir, imid));
            
        end
        fclose(f);
        
    end
end

