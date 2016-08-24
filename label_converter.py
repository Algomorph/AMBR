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

from collections import OrderedDict


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


class Fields(object):
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


def samples_match(samples, offsets, view_names, timestep_threshold):
    source_found = False
    ix_view = 0
    samples_out = []
    while not source_found and ix_view < len(view_names):
        source_sample = samples[ix_view]
        if source_sample is not None:
            source_found = True
        else:
            samples_out.append(None)
        ix_view += 1
    if not source_found:
        raise ValueError("Mouse not in any of the views?")

    source_sample = samples[0]
    source_beginning = source_sample[Fields.beginning] + offsets[0]
    source_end = source_sample[Fields.end] + offsets[0]
    source_label = source_sample[Fields.label]

    for ix_view, target_sample in enumerate(samples[1:]):
        if target_sample is None:
            continue

        target_beginning = target_sample[Fields.beginning] + offsets[ix_view]
        target_end = target_sample[Fields.end] + offsets[ix_view]
        target_label = target_sample[Fields.label]
        if not (abs(source_beginning - target_beginning) < timestep_threshold):
            raise ValueError(
                "Mismatch between sample {:s}({:s}) and {:s}({:s}) beginning times.".format(str(source_sample),
                                                                                            str(target_sample)))
        if not (abs(source_end - target_end) < timestep_threshold):
            raise ValueError(
                "Mismatch between sample {:s}({:s}) and {:s}({:s}) end times.".format(str(source_sample)),
                str(target_sample))
        if source_label != target_label:
            raise ValueError(
                "Mismatch between sample {:s}({:s}) and {:s}({:s}) end times.".format(str(source_sample)))
        offsets[ix_view] += int(round(((source_beginning-target_beginning)+(source_end-target_end))/2))


def main():
    args = process_arguments(Arguments, "text label -> JSON Label Converter for LSTM")
    if args.multiview_only and args.single_view_only:
        raise ValueError("{s} and {s} arguments cannot be combined.".format(Arguments.single_view_only.name,
                                                                            Arguments.multiview_only.name))
    input_paths = []
    view_names = []

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
                view_names.append(base_name)
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
            view_names.append(base_name)

    multiview_samples = {}

    frame_count = 0

    # process each input file one-by-one
    for input_index, (labels_path, feature_path) in enumerate(input_paths):
        input_file = args.input[input_index]
        base_name = view_names[input_index]
        present_flags = np.load(feature_path)

        if frame_count > 0 and frame_count != len(present_flags):
            raise ValueError("Frame counts don't match. Expecting: {:d}. " +
                             "Got {:d} for data source {:s}".format(frame_count, len(present_flags), base_name))
        frame_count = len(present_flags)

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

    if args.single_view_only:
        return 0

    offsets = []
    for key in multiview_samples.keys():
        multiview_samples[key] = {sample[Fields.beginning]: sample for sample in multiview_samples[key]}
        offsets.append(0)

    matched_samples = OrderedDict()
    samples_per_matched_set = len(multiview_samples)
    active_samples = [None] * samples_per_matched_set
    samples_for_matching_count = 0

    ix_frame = 0
    while ix_frame < frame_count:
        for ix_view, view in enumerate(view_names):
            if active_samples[ix_view] is None and ix_frame in multiview_samples[view]:
                if not multiview_samples[view][ix_frame][Fields.label] != label_mapping['X']:
                    active_samples[ix_view] = multiview_samples[view][ix_frame]
                samples_for_matching_count += 1

        if samples_for_matching_count == samples_per_matched_set:
            samples_match(samples)
            active_samples = [None] * len(multiview_samples)
            # ix_frame
        else:
            ix_frame += 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
