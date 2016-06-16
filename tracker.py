#!/usr/bin/python3

import sys
import numpy as np
import calib.io as cio

def main():
    cams = ['al', 'ar', 'cl', 'cr', 'el', 'er', 'dl', 'dr']
    root_path = "/media/algomorph/Data/reco/cap/mouse_cap04"
    data = []
    for cam in cams:
        data.append(np.load(root_path+"/"+cam+"/"+cam+"0_s_data.npz")['frame_data'])


if __name__ == '__main__':
    sys.exit(main())

