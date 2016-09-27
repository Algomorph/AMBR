#!/bin/bash
./extract_vgg_features_from_video.py al.mp4 -mf al_mask.png -m -sis 0 -sie 1800 -bc -nv -aug -s 2100
./extract_vgg_features_from_video.py ar.mp4 -mf ar_mask.png -m -sis 0 -sie 1800 -bc -nv -aug -s 2100
./extract_vgg_features_from_video.py cl.mp4 -mf cl_mask.png -m -sis 0 -sie 1800 -bc -nv -aug -s 2100
./extract_vgg_features_from_video.py cr.mp4 -mf cr_mask.png -m -sis 0 -sie 1800 -bc -nv -aug -s 2100
./extract_vgg_features_from_video.py dl.mp4 -mf dl_mask.png -m -sis 0 -sie 1800 -bc -nv -aug -s 2100
./extract_vgg_features_from_video.py dr.mp4 -mf dr_mask.png -m -sis 0 -sie 1800 -bc -nv -aug -s 2100
./extract_vgg_features_from_video.py el.mp4 -mf el_mask.png -m -sis 0 -sie 1800 -bc -nv -aug -s 2100
./extract_vgg_features_from_video.py er.mp4 -mf er_mask.png -m -sis 0 -sie 1800 -bc -nv -aug -s 2100