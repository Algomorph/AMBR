#!/bin/bash
./extract_vgg_features_from_video.py al0.mp4 -mf al0_mask.png -m -sis 22498 -sie 23396 -bc
./extract_vgg_features_from_video.py ar0.mp4 -mf ar0_mask.png -m -sis 22503 -sie 23399 -bc
./extract_vgg_features_from_video.py cl0.mp4 -mf cl0_mask.png -m -sis 7862 -sie 9056 -bc
./extract_vgg_features_from_video.py cr0.mp4 -mf cr0_mask.png -m -sis 7862 -sie 9095 -bc
./extract_vgg_features_from_video.py dl0.mp4 -mf dl0_mask.png -m -sis 34055 -sie 34664 -bc
./extract_vgg_features_from_video.py dr0.mp4 -mf dr0_mask.png -m -sis 34045 -sie 34664 -bc
./extract_vgg_features_from_video.py el0.mp4 -mf el0_mask.png -m -sis 24655 -sie 26182 -bc
./extract_vgg_features_from_video.py er0.mp4 -mf er0_mask.png -m -sis 24653 -sie 26143 -bc