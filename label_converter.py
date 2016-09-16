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
    input = Argument(arg_help="Label .txt input files. By default, all files in the work folder " +
                              " ending with '_labels.txt' are read.", nargs="+", default=None)
    output_suffix = Argument(arg_help="Label output file suffix.", default=None)
    default_label = Argument(arg_help="Whether to generate default label sequences between the labeled ones.",
                             arg_type='bool_flag', default=False, action='store_true')
    min_sequence_length = Argument(arg_help="Minimum length for each generated sequence.", arg_type=int, default=10)
    time_offset_threshold = Argument(arg_help="Maximum (non-cumulative) frame offset between " +
                                              "corresponding sequences in different views.", arg_type=int, default=10)
    single_view_only = Argument(arg_help="Generate only single view label files.",
                                arg_type='bool_flag', default=False, action='store_true')
    multiview_only = Argument(arg_help="Generate only multiview label file (assumes all " +
                                       "input files are from different views).",
                              arg_type='bool_flag', default=False, action='store_true')


class Fields(object):
    start = "start"
    end = "end"
    label = "label"


label_mapping = {
    "X": 0,
    "G": 1,
    "R": 2,
    "S": 3,
    "C": 4
}

inverse_label_mapping = {
    0: "X",
    1: "G",
    2: "R",
    3: "S",
    4: "C"
}


def generate_sample_dict(beginning_frame, end_frame, label):
    sample = OrderedDict()
    sample[Fields.start] = beginning_frame
    sample[Fields.end] = end_frame
    sample[Fields.label] = label
    return sample


def sample_end(sample):
    if type(sample) == list:
        return sample[-1][Fields.end]
    return sample[Fields.end]


def sample_len(sample):
    if type(sample) == list:
        return sample[-1][Fields.end] - sample[0][Fields.start] + 1
    return sample[Fields.end] - sample[Fields.start] + 1


def clear_path_if_present(path):
    if os.path.exists(path):
        if os.path.isfile(path):
            os.remove(path)
        else:
            shutil.rmtree(path)


def label_at_time(sample_list, time):
    if len(sample_list) == 0:
        return -1, -1
    ix = 0
    while ix < len(sample_list) and not (sample_list[ix][Fields.start] <= time <= sample_list[ix][Fields.end]):
        ix += 1
    if ix == len(sample_list):
        return -1, -1
    return sample_list[ix][Fields.label], ix


def print_matched_set(matched_set, print_endpoints=False):
    change_points = set()
    for sample_list in matched_set:
        for sample in sample_list[1:]:
            change_points.add(sample[Fields.start])
    change_points = list(change_points)
    change_points.sort()

    label_array = np.zeros((len(change_points) + 1, len(matched_set)), dtype=np.int32)
    start_point_array = -np.ones((len(change_points) + 1, len(matched_set)), dtype=np.int32)
    end_point_array = -np.ones((len(change_points) + 1, len(matched_set)), dtype=np.int32)
    border_array = np.zeros((len(change_points) + 1, len(matched_set)), dtype=np.bool)

    current_labels = np.zeros(shape=(len(matched_set),), dtype=np.int32)
    for ix_view, sample_list in enumerate(matched_set):
        if len(sample_list) > 0:
            current_labels[ix_view] = sample_list[0][Fields.label]
            start_point_array[0][ix_view] = sample_list[0][Fields.start]
        else:
            current_labels[ix_view] = -1
    previous_labels = current_labels
    label_array[0] = current_labels

    for ix_step, current_time in enumerate(change_points):
        current_labels = np.zeros(shape=(len(matched_set),), dtype=np.int32)
        for ix_view, sample_list in enumerate(matched_set):
            current_labels[ix_view], ix_sample = label_at_time(sample_list, current_time)
            if current_labels[ix_view] != previous_labels[ix_view]:
                end_point_array[ix_step, ix_view] = -1 if ix_sample < 0 else sample_list[ix_sample - 1][Fields.end]
                border_array[ix_step, ix_view] = True  # previous cells
                start_point_array[ix_step + 1, ix_view] = -1 if ix_sample < 0 else sample_list[ix_sample][Fields.start]

        label_array[ix_step + 1] = current_labels
        previous_labels = current_labels

    for ix_view, sample_list in enumerate(matched_set):
        if len(sample_list) > 0:
            end_point_array[len(change_points), ix_view] = sample_list[-1][Fields.end]

    for ix_view in range(0, len(matched_set)):
        print("|              |", end="")
    print("\n", end="")
    for ix_step in range(0, len(change_points) + 1):
        if print_endpoints:
            for ix_view in range(0, len(matched_set)):
                start = start_point_array[ix_step, ix_view]
                if start >= 0:
                    print("|{:^14}|".format("from: " + str(start)), end="")
                else:
                    print("|              |", end="")
            print("\n", end="")
        for ix_view in range(0, len(matched_set)):
            label = label_array[ix_step, ix_view]
            if label >= 0:
                print("|{:^14}|".format(inverse_label_mapping[label]), end="")
            else:
                print("|       ?      |", end="")
        print("\n", end="")
        if print_endpoints:
            for ix_view in range(0, len(matched_set)):
                end = end_point_array[ix_step, ix_view]
                if end >= 0:
                    print("|{:^14}|".format("to: " + str(end)), end="")
                else:
                    print("|              |", end="")
            print("\n", end="")
        for ix_view in range(0, len(matched_set)):
            if border_array[ix_step, ix_view]:
                print("================", end="")
            else:
                print("|              |", end="")
        print("\n", end="")
    for ix_view in range(0, len(matched_set)):
        print("================", end="")
    print("\n", end="")


def print_multiview_headers(view_names):
    for _ in view_names:
        print("================", end="")
    sys.stdout.write("\n")
    for _ in view_names:
        print("|              |", end="")
    print("\n", end="")
    for name in view_names:
        print("|{:^14}|".format(name), end="")
    print("\n", end="")
    for _ in view_names:
        print("|              |", end="")
    print("\n", end="")
    for _ in view_names:
        print("================", end="")
    print("\n", end="")


def print_appended_line(view_count):
    for i in range(0, view_count):
        print("|      (a)     |", end="")
    print("\n", end="")


def print_mismatch_line(view_count):
    for i in range(0, view_count):
        print("|      (!)     |", end="")
    print("\n", end="")


def print_offsets(offsets):
    for offset in offsets:
        print("|{:^14}|".format("offset: " + str(offset)), end="")
    print("\n", end="")
    for _ in offsets:
        print("================", end="")
    print("\n", end="")


# Prints the entire matched dataset
def print_matched_labels(matched_labels, view_names, print_endpoints=True):
    print_multiview_headers(view_names)
    for matched_set in matched_labels:
        print_matched_set(matched_set, print_endpoints=print_endpoints)


def calculate_view_offset_at(offset_points, timestep):
    if len(offset_points) == 1:
        return offset_points[0][1]
    ix = 0
    while ix < len(offset_points) and not offset_points[ix][0] > timestep:
        ix += 1
    if ix < len(offset_points):
        offset_range = offset_points[ix][0] - offset_points[ix - 1][0]
        ratio = (timestep - offset_points[ix - 1][0]) / offset_range
        return int(round(offset_points[ix - 1][1] * (1.0 - ratio) + offset_points[ix][1] * ratio))
    return offset_points[-1][1]


def main():
    args = process_arguments(Arguments, "text label -> JSON Label Converter for LSTM")
    verbosity = 1
    if args.multiview_only and args.single_view_only:
        raise ValueError("{:s} and {:s} arguments cannot be combined.".format(Arguments.single_view_only.name,
                                                                              Arguments.multiview_only.name))
    input_paths = []
    view_names = []

    x_re = re.compile(r"\[\s*out of visual field\s*\]|\[\s*out of frame\s*\]")
    parse_re = re.compile(
        r"(\d+)\s*(?:\(\?\))?\s*-\s*(\d+)\s*(?:\(\?\))?(?::|;)?\s*(R|G|S|X)\s*?(\?)?\s*(?:\s*\+(\d+))?")

    # determine input file paths
    default_input_file_ending = "_labels.txt"
    if args.input is None or len(args.input) == 0:
        args.input = []
        all_files_in_dir = os.listdir(args.folder)
        all_files_in_dir.sort()
        for file_name in all_files_in_dir:
            file_path = os.path.join(args.folder, file_name)
            if os.path.isfile(file_path) and file_name.endswith(default_input_file_ending):
                args.input.append(file_name)
                view_name = file_name[:-len(default_input_file_ending)]
                feature_file_name = view_name + "_features.npz"
                feature_path = os.path.join(args.folder, feature_file_name)
                if not os.path.isfile(feature_path):
                    raise (IOError("Could not find feature file at {:s}. Each label text file should have a " +
                                   "corresponding feature file.".format(feature_path)))
                input_paths.append((file_path, feature_path))
                view_names.append(view_name)
    else:
        for file_name in args.input:
            file_path = os.path.join(args.folder, file_name)
            if not os.path.isfile(file_path):
                raise IOError("Could not find file at {:s}".format(file_path))
            if file_name.endswith(default_input_file_ending):
                view_name = file_name[:-len(default_input_file_ending)]
                feature_file_name = view_name + "_features.npz"
            else:
                view_name = file_name[:-4]
                feature_file_name = view_name + "_features.npz"
            feature_path = os.path.join(args.folder, feature_file_name)
            if not os.path.isfile(feature_path):
                raise (IOError("Cound not find feature file at {:s}. Each label text file should have a " +
                               "corresponding feature file.".format(feature_path)))
            input_paths.append((file_path, feature_path))
            view_names.append(view_name)

    multiview_samples = []
    view_offsets = []

    frame_count = sys.maxsize

    # process each input file one-by-one
    for input_index, (labels_path, feature_path) in enumerate(input_paths):
        input_file = args.input[input_index]
        if verbosity > 0:
            print("Processing file {:s}".format(input_file))
        view_name = view_names[input_index]
        present_flags = np.load(feature_path)['present']

        if frame_count > 0 and frame_count != sys.maxsize and frame_count != len(present_flags):
            print(("WARNING: Frame counts don't match. Expecting: {:d}. " +
                   "Got {:d} for data source {:s}").format(frame_count, len(present_flags), view_name))
        if frame_count > len(present_flags):
            frame_count = len(present_flags)

        if not args.multiview_only:
            if args.output_suffix:
                output_file = input_file[:-4] + args.output_suffix + ".json"
            else:
                output_file = input_file[:-3] + "json"
            output_path = os.path.join(args.folder, output_file)

        file_handle = open(labels_path, 'r')
        lines = tuple(file_handle)
        file_handle.close()

        default_behavior_label = 4
        samples = []

        offset_points = [(0, 0)]

        # parse the label file
        for line in lines:
            line = x_re.sub("X", line, re.IGNORECASE)
            match_range = parse_re.match(line)
            if match_range:
                if match_range.group(4) is None:
                    start = int(match_range.group(1))
                    end = int(match_range.group(2))
                    label = label_mapping[match_range.group(3)]
                    if label != 0:  # drop manual X labels; we will read them from the features file
                        samples.append(generate_sample_dict(start, end, label))
                if match_range.group(5) is not None:
                    offset_point = int(match_range.group(5))
                    offset_points.append((start, offset_point))
            else:
                print("Unmatched line: {:s}".format(line))
        if verbosity >= 1:
            print("Line count: {:d}\nSamples parsed: {:d}".format(len(lines), len(samples)))

        # insert automatic "out of frame / X labels"
        samples_processed = []
        start_at = 0

        for sample in samples + [generate_sample_dict(len(present_flags), 0, -99)]:
            if sample[Fields.start] > 0:
                begin_x_frame = start_at
                while begin_x_frame < sample[Fields.start]:
                    if not present_flags[begin_x_frame]:
                        end_x_frame = begin_x_frame
                        while end_x_frame < sample[Fields.start] and not present_flags[end_x_frame]:
                            end_x_frame += 1
                        if end_x_frame - begin_x_frame < args.min_sequence_length:
                            if verbosity >= 2:
                                print(("WARNING: Got a suspiciously short X (out of frame) sequence,"
                                       + " [{:d},{:d}], for source {:s}")
                                      .format(begin_x_frame, end_x_frame - 1, view_name))
                        else:
                            samples_processed.append(
                                generate_sample_dict(begin_x_frame, end_x_frame - 1, label_mapping['X']))
                        begin_x_frame = end_x_frame
                    begin_x_frame += 1
                if sample[Fields.label] != -99:
                    samples_processed.append(sample)
            start_at = sample[Fields.end] + 1
        samples = samples_processed

        # insert "Default" label if required
        # both bounds of each sequence are inclusive
        if args.default_label:
            samples_processed = []
            if args.default_label and samples[0][Fields.start] > args.min_sequence_length:
                samples_processed.append(
                    generate_sample_dict(0, samples[0][Fields.start] - 1, default_behavior_label))
            for ix_sample in range(len(samples) - 1):
                sample = samples[ix_sample]
                next_sample = samples[ix_sample + 1]
                samples_processed.append(sample)
                if next_sample[Fields.start] - (sample[Fields.end] + 1) > args.min_sequence_length:
                    samples_processed.append(
                        generate_sample_dict(sample[Fields.end] + 1, next_sample[Fields.start] - 1,
                                             default_behavior_label))
            sample = samples[-1]
            samples_processed.append(sample)
            samples = samples_processed

        if not args.multiview_only:
            if verbosity > 0:
                print("Saving output to {:s}".format(output_file))
            file_handle = open(output_path, 'w')
            json.dump(samples, file_handle, indent=3)
            file_handle.close()

        multiview_samples.append(samples)
        view_offsets.append(offset_points)

    if args.single_view_only:
        return 0

    multiview_hashes = [{sample[Fields.start]: sample for sample in multiview_sample_set} for multiview_sample_set in
                        multiview_samples]
    view_count = len(view_names)

    source_view = view_names[0]

    # go through all sample sets one-by-one, treating them as "source"
    # looks only for matches between non-default label segments here
    nondefault_sample_groups = []
    nondefault_processed = [set() for _ in view_names]
    for ix_source_view, source_view in enumerate(view_names):
        nondefault_processed_source = nondefault_processed[ix_source_view]
        for source_sample in multiview_samples[ix_source_view]:
            if source_sample[Fields.label] == default_behavior_label or source_sample[Fields.label] == label_mapping[
                'X'] \
                    or source_sample[Fields.start] in nondefault_processed_source:
                continue
            sample_group = [None] * view_count
            sample_group[ix_source_view] = [source_sample]
            for ix_target_view in range(0, view_count):
                if ix_target_view == ix_source_view:
                    continue
                closest_time_gap = sys.maxsize
                closest_sample = None
                for target_sample in multiview_samples[ix_target_view]:
                    if (target_sample[Fields.label] == default_behavior_label or (
                                    target_sample[Fields.label] != label_mapping['X'] and
                                    target_sample[Fields.label] != source_sample[Fields.label])
                        or target_sample[Fields.start] in nondefault_processed[ix_target_view]):
                        continue
                    time_gap = abs(source_sample[Fields.start] - target_sample[Fields.start])
                    # if the mouse is not in view for sure during that source sequence, match that not-in-view sequence
                    if (target_sample[Fields.label] == label_mapping['X'] and
                            (source_sample[Fields.start] - target_sample[Fields.start] > args.time_offset_threshold
                             and target_sample[Fields.end] - source_sample[Fields.end] > args.time_offset_threshold)):
                        closest_sample = target_sample
                        break

                    if closest_time_gap > time_gap:
                        closest_time_gap = time_gap
                        closest_sample = target_sample
                # didn't find a "closest" sample at all!
                if closest_sample is None:
                    raise (ValueError(
                        "Could not find match for samples {:s}:{:s} and {:s}:{:s}".format(str(source_sample),
                                                                                          source_view,
                                                                                          str(target_sample),
                                                                                          view_names[ix_target_view])))
                if closest_sample[Fields.label] != label_mapping['X'] and closest_time_gap > args.time_offset_threshold:
                    raise (ValueError(
                        "Starting times for samples {:s}:{:s} and {:s}:{:s} are too far apart. Gap: {:d}".format(
                            str(source_sample), source_view,
                            str(closest_sample), view_names[ix_target_view],
                            closest_time_gap)))
                if closest_sample[Fields.label] != label_mapping['X']:
                    nondefault_processed[ix_target_view].add(closest_sample[Fields.start])
                sample_group[ix_target_view] = [closest_sample]
            nondefault_sample_groups.append(sample_group)
            nondefault_processed_source.add(source_sample[Fields.start])

    nondefault_sample_groups.sort(
        key=lambda sample_group: np.min(
            [view_list[0][Fields.start] if view_list[0][Fields.label] != label_mapping['X'] else sys.maxsize for
             view_list in sample_group]))

    # now, look into the time-gaps between our matched sets
    sample_groups = []
    previous_end_times = [0 for _ in view_names]
    for nondefault_sample_group in nondefault_sample_groups:
        default_or_empty_sample_group = [[] for _ in view_names]
        for ix_view, nondefault_sample_set in enumerate(nondefault_sample_group):
            prev_end_time = previous_end_times[ix_view]
            nondefault_sample = nondefault_sample_set[0]
            next_start_time = nondefault_sample[Fields.start]
            span = next_start_time - prev_end_time + 1
            view_sample_hash = multiview_hashes[ix_view]
            if span >= args.min_sequence_length:
                for ix_frame in range(prev_end_time + 1, next_start_time):
                    if ix_frame in view_sample_hash:
                        sample = view_sample_hash[ix_frame]
                        if sample[Fields.label] == label_mapping['X']:
                            continue
                        if sample[Fields.label] == default_behavior_label:
                            default_or_empty_sample_group[ix_view].append(sample)
                        else:
                            raise (ValueError("Unmatched sample {:s} for view {:s}"
                                              .format(str(sample), view_names[ix_view])))

            previous_end_times[ix_view] = nondefault_sample[Fields.end]
            if nondefault_sample[Fields.label] == label_mapping['X']:
                nondefault_sample_group[ix_view] = []
        sample_groups.append(default_or_empty_sample_group)
        sample_groups.append(nondefault_sample_group)

    print_matched_labels(sample_groups, view_names)

    output_path = os.path.join(args.folder, "multiview_samples.json")
    file_handle = open(output_path, 'w')
    json.dump(sample_groups, file_handle, indent=3)
    file_handle.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
