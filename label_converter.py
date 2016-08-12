#!/usr/bin/python3
"""
text label -> JSON Label Converter for LSTM
authors - Chethan Mysore Parameshwara, Greg Kramida
emails - analogicalnexus159@gmail.com, algomorph@gmail.com
"""
import json
import re
import sys
import numpy as np
import os.path
import os
import argparse as ap
import shutil

parser = ap.ArgumentParser("text label -> JSON Label Converter for LSTM")
parser.add_argument("-f", "--folder", type=str, default="./")
parser.add_argument("-i", "--input_file", type=str, default="al_labels.txt")
parser.add_argument("-o", "--output_file", type=str, default=None)
parser.add_argument("-dl", "--default_label", default=False, action='store_true',
                    help="Whether to use default label or not.")


def generate_sample_hash(s_fid, e_fid, label):
    data = {
        'start': int(s_fid),
        'end': int(e_fid),
        'label': int(label)
    }
    return data

# writer = csv.writer(open("label.csv", "w"), delimiter=',', lineterminator='\n')
def main():
    args = parser.parse_args()
    if args.output_file is None:
        args.output_file = os.path.join(os.path.dirname(args.input_file),
                                        os.path.basename(args.input_file)[:-3] + "json")

    args.input_file = os.path.join(args.folder, args.input_file)
    args.output_file = os.path.join(args.folder, args.output_file)

    if os.path.exists(args.output_file):
        if os.path.isfile(args.output_file):
            os.remove(args.output_file)
        else:
            shutil.rmtree(args.output_file)

    lines = tuple(open(args.input_file, 'r'))
    x_re = re.compile(r"\[\s*out of visual field\s*\]|\[\s*out of frame\s*\]")
    parse_re = re.compile(r"(\d+)\s*(?:\(\?\))?\s*-\s*(\d+)\s*(?:\(\?\))?(?::|;)?\s*(R|G|S|X)\s*?(\?)?\s*")
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
        line = x_re.sub("X", line, re.IGNORECASE)
        match = parse_re.match(line)
        if match:
            if match.group(4) is None:
                start = int(match.group(1))
                end = int(match.group(2))
                label = label_mapping[match.group(3)]
                sample_data.append([start, end, label])
        else:
            print("Unmatched line: {:s}".format(line))
    print("num lines: {:d}\nnum matches: {:d}".format(len(lines), len(sample_data)))

    sample_data = np.array(sample_data, dtype=np.int32)
    data_out = []

    if args.default_label and sample_data[0, 0] > 0:
        data_out.append(generate_sample_hash(0, sample_data[0, 0] - 1, default_behavior_label))
    for ix_sample in range(len(sample_data) - 1):
        sample = sample_data[ix_sample]
        next_sample = sample_data[ix_sample + 1]
        data_out.append(generate_sample_hash(sample[0], sample[1], sample[2]))
        if args.default_label and sample[1] + 1 != next_sample[0]:
            data_out.append(generate_sample_hash(sample[1] + 1, next_sample[0] - 1, default_behavior_label))
    sample = sample_data[-1]
    data_out.append(generate_sample_hash(sample[0], sample[1], sample[2]))
    with open(args.output_file, 'a') as file:
        json.dump(data_out, file, indent=3)

    return 0


if __name__ == "__main__":
    sys.exit(main())
