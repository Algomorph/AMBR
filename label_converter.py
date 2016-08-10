"""
text label -> JSON Label Converter for LSTM
authors - Chethan Mysore Parameshwara, Greg Kramida
emails - analogicalnexus159@gmail.com, algomorph@gmail.com
"""
import json
import re
import sys
import numpy as np


def csv2json(s_fid, e_fid, label):
    data = {
        's_fid': s_fid,
        'e_fid': e_fid,
        'label': label
    }
    return data


def jsonwrite(file, s_fid, e_fid, label):
    data = csv2json(s_fid, e_fid, label)
    json.dump(data, file, indent=3)
    file.write(',')
    file.write('\n')


# writer = csv.writer(open("label.csv", "w"), delimiter=',', lineterminator='\n')
def main():
    #TODO: make input & output options
    lines = tuple(open("al0_labels.txt", 'r'))
    parse_re = re.compile(r"(\d+)\s*-(\d+):\s*(R|G|S|\[out of frame\])\s*?(\?)\s*?")
    label_mapping = {
        "X": 0,
        "[out of frame]": 0,
        "G": 1,
        "R": 2,
        "S": 3,
    }
    default_behavior_label = 4
    sample_data = []
    for line in lines:
        match = parse_re.match(line)
        if match:
            if match.group(4) is None:
                start = int(match.group(0))
                end = int(match.group(1))
                label = label_mapping[match.group(2)]
            sample_data.append([start, end, label])

    sample_data = np.array(sample_data, dtype=np.int32)
    with open('al0_wo5_labels.json', 'a') as file:
        if sample_data[0, 0] > 0:
            jsonwrite(0, sample_data[0, 0] - 1, default_behavior_label)
        for ix_sample in len(sample_data)-1:
            sample = sample_data[ix_sample]
            next_sample = sample_data[ix_sample+1]
            jsonwrite(sample[0], sample[1], sample[2])
            #TODO: make this an option
            #TODO: add checking for consecutive intervals
            jsonwrite(sample[1] + 1, next_sample[0] - 1, default_behavior_label)
    return 0


if __name__ == "__main__":
    sys.exit(main())
