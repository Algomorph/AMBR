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
import shutil
from enum import Enum

# local:
from ext_argparse.argproc import process_arguments
from ext_argparse.argument import Argument


class Arguments(Enum):
    folder = Argument(arg_help="Folder to work in.", setting_file_location=True, default=".")
    input = Argument(shorthand="-i", arg_help="Label .txt input files. By default, all files in the work folder " +
                                              " ending with '_labels.txt' are read.", nargs="+", default=None)
    output_suffix = Argument(arg_help="Label output file suffix.", default=None)
    default_label = Argument(arg_help="Whether to generate default label sequences between the labeled ones.",
                             arg_type='bool_flag', default=False, action='store_true')
    min_seq_length = Argument(arg_help="Minimum length for each generated sequence.", arg_type=int, default=10)
    single_view_only = Argument(arg_help="Generate only single view label files.",
                                arg_type='bool_flag', default=False, action='store_true')
    multiview_only = Argument(arg_help="Generate only multiview label file (assumes all " +
                                       "input files are from different views).",
                              arg_type='bool_flag', default=False, action='store_true')


class Fields():
    beginning = "beginning"
    end = "end"
    label = "label"


def generate_sample_dict(beginning_frame, end_frame, label):
    sample = {
        Fields.beginning: int(beginning_frame),
        Fields.end: int(end_frame),
        Fields.label: int(label)
    }
    return sample


# writer = csv.writer(open("label.csv", "w"), delimiter=',', lineterminator='\n')
def main():
    args = process_arguments(Arguments, "text label -> JSON Label Converter for LSTM")
    if args.multiview_only and args.single_view_only:
        raise ValueError("{s} and {s} arguments cannot be combined.".format(Arguments.single_view_only.name,
                                                                            Arguments.multiview_only.name))
    input_paths = []
    base_names = []

    x_re = re.compile(r"\[\s*out of visual field\s*\]|\[\s*out of frame\s*\]")
    parse_re = re.compile(r"(\d+)\s*(?:\(\?\))?\s*-\s*(\d+)\s*(?:\(\?\))?(?::|;)?\s*(R|G|S|X)\s*?(\?)?\s*")

    # determine input file paths
    default_input_file_ending = "_labels.txt"
    if args.input is None or len(args.input) == 0:
        args.input = []
        all_files_in_dir = os.listdir(args.folder)
        for file_name in all_files_in_dir:
            file_path = os.path.join(args.folder, file_name)
            args.input.append(file_name)
            if os.path.isfile(file_path) and file_name.endswith(default_input_file_ending):
                base_name = file_name[:-len(default_input_file_ending)]
                feature_file_name = base_name + "_features.npz"
                feature_path = os.path.join(args.folder, feature_file_name)
                if not os.path.isfile(feature_path):
                    raise (IOError("Cound not find feature file at {:s}. Each label text file should have a " +
                                   "corresponding feature file.".format(feature_path)))
                input_paths.append((file_path, feature_path))
                base_names.append(base_name)
    else:
        for file_name in args.input:
            file_path = os.path.join(args.folder, file_name)
            if not os.path.isfile(file_path):
                raise IOError("Could not find file at {:s}".format(file_path))
            if file_name.endswith(default_input_file_ending):
                base_name = file_name[:-len(default_input_file_ending)]
                feature_file_name = base_name + "_features.npz"
            else:
                base_name = file_name[:-4]
                feature_file_name = base_name + "_features.npz"
            feature_path = os.path.join(args.folder, feature_file_name)
            if not os.path.isfile(feature_path):
                raise (IOError("Cound not find feature file at {:s}. Each label text file should have a " +
                               "corresponding feature file.".format(feature_path)))
            input_paths.append((file_path, feature_path))
            base_names.append(base_name)

    multiview_samples = {}

    # process each input file one-by-one
    for input_index, (labels_path, feature_path) in enumerate(input_paths):
        input_file = args.input[input_index]
        base_name = base_names[input_index]
        present_flags = np.load(feature_path)

        if args.output_suffix:
            output_file = input_file[:-4] + args.output_suffix + ".json"
        else:
            output_file = input_file[:-3] + "json"

        # erase the output file if we're going to save one for each input and there's already an output file there
        if not args.multiview_only:
            output_path = os.path.join(args.folder, output_file)
            if os.path.exists(output_path):
                if os.path.isfile(output_path):
                    os.remove(args.output_file)
                else:
                    shutil.rmtree(output_path)

        file_handle = open(labels_path, 'r')
        lines = tuple(file_handle)
        file_handle.close()

        label_mapping = {
            "X": 0,
            "G": 1,
            "R": 2,
            "S": 3,
        }
        default_behavior_label = 4
        samples = []

        # parse the label file
        for line in lines:
            line = x_re.sub("X", line, re.IGNORECASE)
            match = parse_re.match(line)
            if match:
                if match.group(4) is None:
                    beginning = int(match.group(1))
                    end = int(match.group(2))
                    label = label_mapping[match.group(3)]
                    if label != 0:  # drop manual X labels; we will read them from the features file
                        samples.append(generate_sample_dict(beginning, end, label))

            else:
                print("Unmatched line: {:s}".format(line))
        print("num lines: {:d}\nnum matches: {:d}".format(len(lines), len(samples)))

        samples = np.array(samples, dtype=np.int32)
        # insert automatic "out of frame / X labels"
        samples_processed = []
        start_at = 0

        for sample in samples + [generate_sample_dict(len(present_flags), 0, -99)]:
            if sample[Fields.beginning] > 0:
                begin_x_frame = start_at
                while begin_x_frame < sample[Fields.beginning]:
                    if not present_flags[begin_x_frame]:
                        end_x_frame = begin_x_frame
                        while end_x_frame < sample[Fields.beginning] and not present_flags[end_x_frame]:
                            end_x_frame += 1
                        if end_x_frame - begin_x_frame < args.min_sequence_length:
                            raise ValueError("Got a suspiciously short X (out of frame) sequence, [{:d},{:d}]"
                                             .format(begin_x_frame, end_x_frame - 1))
                        samples_processed.append(
                            generate_sample_dict(begin_x_frame, end_x_frame - 1, label_mapping['X']))
                        begin_x_frame = end_x_frame
                    begin_x_frame += 1
                if sample[Fields.label] != -99:
                    samples_processed.append(sample)
            start_at = sample[Fields.end] + 1

        # insert "Default" label if required
        # both bounds of each sequence are inclusive
        if args.default_label:
            samples_processed = []
            if args.default_label and samples[0][Fields.beginning] > args.min_sequence_length:
                samples_processed.append(
                    generate_sample_dict(0, samples[0][Fields.beginning] - 1, default_behavior_label))
            for ix_sample in range(len(samples) - 1):
                sample = samples[ix_sample]
                next_sample = samples[ix_sample + 1]
                samples_processed.append(sample)
                if next_sample[Fields.beginning] - (sample[Fields.end] + 1) > args.min_sequence_length:
                    samples_processed.append(
                        generate_sample_dict(sample[Fields.end] + 1, next_sample[Fields.beginning] - 1,
                                             default_behavior_label))
            sample = samples[-1]
            samples_processed.append(sample)
            samples = samples_processed

        if not args.multiview_only:
            file_handle = open(args.output_file, 'a')
            json.dump(samples, file_handle, indent=3)
            file_handle.close()

        multiview_samples[base_name] = samples

    if not args.single_view_only:
        # TODO
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
