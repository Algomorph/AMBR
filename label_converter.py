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
    multiview_format = Argument(arg_help="Use multiview label input file format (input arg required)",
                                arg_type='bool_flag', default=False, action='store_true')
    verbosity = Argument(arg_help="Verbosity level", arg_type=int, default=1)
    global_offset = Argument(arg_help="Global offset (with respect to the features) to add to all the frame numbers.",
                             arg_type=int, default=0)


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


def generate_sample_dict(start_frame, end_frame, label):
    sample = OrderedDict()
    sample[Fields.start] = start_frame
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


def insert_out_of_frame_sequences(args, samples, present_flags, view_name):
    """
    Takes samples w/o X (not-in-frame) labels for a single view and the per-frame "present" flag array for that view
    and generates the missing X sequence samples for those intervals whose frames are marked as "false" in the "present"
    flag array.
    :param args: program args
    :param samples: array of sequence samples with no "empty/out-of-frame" (X) labels
    :param present_flags: array of flags indicating whether the subject is present in the sequence or not
    :param view_name: name of the current view
    :return: processed samples array with the new X-labeled sequences
    """
    samples_processed = []
    start_at = 0

    for sample in samples + [generate_sample_dict(len(present_flags), 0, -99)]:
        if sample[Fields.start] > 0:
            start_x_frame = start_at
            while start_x_frame < sample[Fields.start]:
                if start_x_frame == len(present_flags):
                    print(sample)
                if not present_flags[start_x_frame]:
                    end_x_frame = start_x_frame
                    while end_x_frame < sample[Fields.start] and not present_flags[end_x_frame]:
                        end_x_frame += 1
                    if end_x_frame - start_x_frame < args.min_sequence_length:
                        if args.verbosity >= 2:
                            print(("WARNING: Got a suspiciously short X (out of frame) sequence,"
                                   + " [{:d},{:d}], for source {:s}")
                                  .format(start_x_frame, end_x_frame - 1, view_name))
                    else:
                        samples_processed.append(
                            generate_sample_dict(start_x_frame,
                                                 end_x_frame - 1, label_mapping['X']))
                    start_x_frame = end_x_frame
                start_x_frame += 1
            if sample[Fields.label] != -99:
                samples_processed.append(sample)
        start_at = sample[Fields.end] + 1

    return samples_processed


def insert_default_label_sequences(args, samples, default_behavior_label):
    """
    Takes sample sequence list for a single view and inserts all the missing sequences in-between, marking them
    with "default label"
    :param args: program args
    :param samples: sample sequence list for a single view w/o default labels
    :param default_behavior_label: the numeric version of the default label
    :return: (ordered) sample list with the original sequences and the "default"-label sequences in-between them.
    """
    samples_processed = []
    if args.default_label and samples[0][Fields.start] > args.min_sequence_length:
        samples_processed.append(
            generate_sample_dict(0, samples[0][Fields.start] - 1,
                                 default_behavior_label))
    for ix_sample in range(len(samples) - 1):
        sample = samples[ix_sample]
        next_sample = samples[ix_sample + 1]
        samples_processed.append(sample)
        if next_sample[Fields.start] - (sample[Fields.end]) > args.min_sequence_length:
            samples_processed.append(
                generate_sample_dict(sample[Fields.end] + 1,
                                     next_sample[Fields.start] - 1,
                                     default_behavior_label))
    sample = samples[-1]
    samples_processed.append(sample)
    return samples_processed


def calculate_sample_overlap(sample_a, sample_b):
    if sample_a[Fields.end] < sample_b[Fields.start] or sample_a[Fields.start] > sample_b[Fields.end]:
        return 0
    if sample_a[Fields.end] > sample_b[Fields.end]:
        if sample_a[Fields.start] <= sample_b[Fields.start]:
            # full overlap of b
            return sample_b[Fields.end] + 1 - sample_b[Fields.start]
        else:
            return sample_b[Fields.end] + 1 - sample_a[Fields.start]
    else:
        if sample_b[Fields.start] <= sample_a[Fields.start]:
            # full overlap of a
            return sample_a[Fields.end] + 1 - sample_a[Fields.start]
        else:
            return sample_a[Fields.end] + 1 - sample_b[Fields.start]


def parse_label_lines(args, lines, multiview_mode=False):
    """
    Parse lines of raw label.txt file received from a human annotator.
    :param args:
    :param lines:
    :param multiview_mode:
    :return:
    """
    x_re = re.compile(r"\[\s*out of visual field\s*\]|\[\s*out of frame\s*\]")
    range_re = re.compile(
        r"(\d+)\s*(?:\(\?\))?\s*-\s*(\d+)\s*(?:\(\?\))?(?::|;)?\s*(R|G|S|X)\s*?(\?)?\s*(?:\s*\+(\d+))?")
    multiview_offset_match_re = re.compile(r"!(?:\s*\w+\s*[+-]\s*\d+,\s*)*(?:\w+\s*[+-]\s*\d+\s*)")
    multiview_offset_parse_re = re.compile(r"\s*(\w+)\s*([+-]\s*\d+)")

    samples = []

    if multiview_mode:
        offset_points = {}
    else:
        offset_points = [(0, 0)]

    last_start = 0

    # parse the label file
    for line in lines:
        line = x_re.sub("X", line, re.IGNORECASE)
        match_range = range_re.match(line)
        if match_range:
            if match_range.group(4) is None:
                start = int(match_range.group(1))
                end = int(match_range.group(2))
                label = label_mapping[match_range.group(3)]
                if label != 0:  # drop manual X labels; we will read them from the features file
                    samples.append(generate_sample_dict(start + args.global_offset, end + args.global_offset, label))
            if not multiview_mode and match_range.group(5) is not None:
                offset = int(match_range.group(5))
                offset_points.append((start + args.global_offset, offset))
            last_start = start
        elif multiview_mode and multiview_offset_match_re.match(line):
            for view_name, offset in multiview_offset_parse_re.findall(line):
                if view_name in offset_points:
                    view_offset_points = offset_points[view_name]
                else:
                    view_offset_points = []
                    offset_points[view_name] = view_offset_points
                view_offset_points.append((last_start + args.global_offset, int(offset)))
        else:
            print("Unmatched line: {:s}".format(line))
    if args.verbosity >= 1:
        print("Line count: {:d}\nSamples parsed: {:d}".format(len(lines), len(samples)))
    return samples, offset_points


def process_individual_label_files(args, default_behavior_label):
    input_paths = []
    view_names = []

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
        if args.verbosity > 0:
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

        samples, offset_points = parse_label_lines(args, lines)

        # ========================= insert automatic "out of frame / X" label sequences ============================== #
        samples = insert_out_of_frame_sequences(args, samples, present_flags, view_name)

        # ========================= insert "Default" label sequences if required ===================================== #
        # both bounds of each sequence are inclusive
        if args.default_label:
            samples = insert_default_label_sequences(args, samples, default_behavior_label)

        # if given multiview-only setting, skip writing json label files for individual views
        if not args.multiview_only:
            if args.verbosity > 0:
                print("Saving output to {:s}".format(output_file))
            file_handle = open(output_path, 'w')
            json.dump(samples, file_handle, indent=3)
            file_handle.close()

        multiview_samples.append(samples)
        view_offsets.append(offset_points)

    return multiview_samples, view_names


def output_and_print_multiview_groups(args, sample_groups, view_names):
    print_matched_labels(sample_groups, view_names)

    output_path = os.path.join(args.folder, "multiview_samples.json")
    file_handle = open(output_path, 'w')
    json.dump(sample_groups, file_handle, indent=3)
    file_handle.close()


def generate_multiview_labels(args, multiview_samples, view_names, default_behavior_label):
    multiview_hashes = [{sample[Fields.start]: sample for sample in multiview_sample_set} for multiview_sample_set in
                        multiview_samples]
    view_count = len(view_names)
    # go through all view sample sets one-by-one, treating each view as "source"
    # looks only for matches between non-default label segments here
    nondefault_sample_groups = []
    nondefault_processed = [set() for _ in view_names]

    for ix_source_view, source_view in enumerate(view_names):
        nondefault_processed_source = nondefault_processed[ix_source_view]
        last_group = None
        for source_sample in multiview_samples[ix_source_view]:
            if source_sample[Fields.label] == default_behavior_label or source_sample[Fields.label] == label_mapping[
                'X'] \
                    or source_sample[Fields.start] in nondefault_processed_source:
                continue
            sample_group = [None] * view_count
            sample_group[ix_source_view] = [source_sample]
            complete_group = True
            for ix_target_view in range(0, view_count):
                if ix_target_view == ix_source_view:
                    continue
                closest_time_gap = sys.maxsize
                closest_sample = None
                # closest_overlap = 0
                missing_view_ixs = []
                for target_sample in multiview_samples[ix_target_view]:
                    if (target_sample[Fields.label] == default_behavior_label or
                                target_sample[Fields.start] in nondefault_processed[ix_target_view] or
                            (target_sample[Fields.label] != label_mapping['X'] and
                                     target_sample[Fields.label] != source_sample[Fields.label])):
                        continue
                    time_gap = abs(source_sample[Fields.start] - target_sample[Fields.start])
                    target_overlap = calculate_sample_overlap(target_sample, source_sample)
                    # if the mouse is not in view for sure during that source sequence, match that not-in-view sequence
                    if (target_sample[Fields.label] == label_mapping['X'] and
                            (source_sample[Fields.start] - target_sample[Fields.start] > args.time_offset_threshold
                             and target_sample[Fields.end] - source_sample[Fields.end] > args.time_offset_threshold)):
                        closest_sample = target_sample
                        break
                    if closest_time_gap > time_gap or (
                                    target_sample[Fields.label] == source_sample[
                                    Fields.label] and target_overlap >= min(sample_len(source_sample) / 2,
                                                                            sample_len(target_sample) / 2)):
                        closest_time_gap = time_gap
                        closest_sample = target_sample
                        # closest_overlap = target_overlap
                if (closest_sample is not None and
                            calculate_sample_overlap(closest_sample, source_sample) >= args.min_sequence_length
                    and ((closest_sample[Fields.label] == label_mapping['X'] or
                                  closest_time_gap <= args.time_offset_threshold))):
                    if closest_sample[Fields.label] != label_mapping['X']:
                        nondefault_processed[ix_target_view].add(closest_sample[Fields.start])
                    sample_group[ix_target_view] = [closest_sample]
                else:
                    if complete_group:
                        complete_group = False
                        bad_sample = closest_sample
                        bad_time_gap = closest_time_gap
                        bad_view = ix_target_view
                    missing_view_ixs.append(ix_target_view)

            if not complete_group and last_group is not None:
                # check if there is significant overlap with the samples in previous group
                complete_group = True
                for ix_view, sample_list in enumerate(sample_group):
                    if sample_list is not None:
                        for ix in missing_view_ixs:
                            if calculate_sample_overlap(sample_list[0], last_group[ix][-1]) < args.min_sequence_length \
                                    and (sample_list[0][Fields.label] != last_group[ix][-1][Fields.label]
                                         or sample_list[0][Fields.label] == 'X'
                                         or last_group[ix][-1][Fields.label] == 'X'):
                                complete_group = False
                            else:
                                last_group[ix_view].append(sample_list[0])
                                if sample_list[0][Fields.label] != label_mapping['X']:
                                    nondefault_processed[ix_view].add(sample_list[0][Fields.start])
            else:
                for sample_list in sample_group:
                    if sample_list is None:
                        print(last_group)
                        raise ValueError("OMG {:s}".format(str(sample_group)))
                last_group = sample_group
                nondefault_sample_groups.append(sample_group)
                nondefault_processed_source.add(source_sample[Fields.start])
            if not complete_group:
                if bad_sample is None:
                    raise (ValueError(
                        "Could not find match for sample {:s}:{:s} in view {:s}".format(str(source_sample),
                                                                                        source_view,
                                                                                        view_names[
                                                                                            ix_target_view])))

                if bad_sample[Fields.label] != label_mapping['X'] and bad_time_gap > args.time_offset_threshold:
                    raise (ValueError(
                        "Starting times for samples {:s}:{:s} and {:s}:{:s} are too far apart. Gap: {:d}".format(
                            str(source_sample), source_view,
                            str(bad_sample), view_names[bad_view], bad_time_gap)))

    # sort groups by start of the first sample in each set
    nondefault_sample_groups.sort(
        key=lambda sample_group: np.min(
            [view_list[0][Fields.start] if view_list[0][Fields.label] != label_mapping['X'] else sys.maxsize for
             view_list in sample_group]))
    print_matched_labels(nondefault_sample_groups, view_names)

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

            previous_end_times[ix_view] = nondefault_sample_set[-1][Fields.end]
            nondefault_sample_group[ix_view] = \
                [sample for sample in nondefault_sample_group[ix_view]
                 if sample[ix_view][Fields.label] != label_mapping['X']]
        sample_groups.append(default_or_empty_sample_group)
        sample_groups.append(nondefault_sample_group)

    output_and_print_multiview_groups(args, sample_groups, view_names)


def duplicate_sample_list(samples):
    duplicate_samples = []
    for sample in samples:
        duplicate_samples.append(generate_sample_dict(sample[Fields.start],
                                                      sample[Fields.end], sample[Fields.label]))
    return duplicate_samples


def process_multiview_offsets(multiview_samples, offset_points, view_names):
    multiview_samples_processed = []
    for view_name, samples in zip(view_names, multiview_samples):
        view_samples = []
        if view_name in offset_points:
            view_offset_points = offset_points[view_name]
            offset_period_start = 0
            offset_start = 0
            i_offset_point = 0
            offset_period_end = view_offset_points[i_offset_point][0]
            offset_end = view_offset_points[i_offset_point][1]
            no_more_offsets = False
            offset = 0
            for sample in samples:
                cur_start = sample[Fields.start]
                if not no_more_offsets:
                    if cur_start > offset_period_end:
                        offset_period_start = offset_period_end
                        offset_start = offset_end
                        if i_offset_point < len(view_offset_points):
                            no_more_offsets = True
                            offset_period_end = offset_period_start + 100000000
                        else:
                            i_offset_point += 1
                            offset_period_end = view_offset_points[i_offset_point][0]
                            offset_end = view_offset_points[i_offset_point][1]

                    ratio = (cur_start - offset_period_start) / (offset_period_end - offset_period_start)
                    offset = round(offset_start * (1.0 - ratio) + offset_end * ratio)
                view_samples.append(
                    generate_sample_dict(sample[Fields.start] + offset,
                                         sample[Fields.end] + offset,
                                         sample[Fields.label]))
        else:
            view_samples = samples
        multiview_samples_processed.append(view_samples)
    return multiview_samples_processed


def stitch_samples(args, samples):
    """
    Stitches together samples that have same label and intervals shorter than args.min_sequence_length
    :param args: program arguments
    :param samples: an ordered set of sequence samples
    :return:
    """
    processed_samples = []
    stitch_sample_group = [samples[0]]
    for sample in samples[1:]:
        if sample[Fields.start] - stitch_sample_group[-1][Fields.end] - 1 < args.min_sequence_length \
                and sample[Fields.label] == stitch_sample_group[0][Fields.label]:
            stitch_sample_group.append(sample)
        else:
            processed_samples.append(
                generate_sample_dict(stitch_sample_group[0][Fields.start], stitch_sample_group[-1][Fields.end],
                                     stitch_sample_group[0][Fields.label]))
            stitch_sample_group = [sample]

    processed_samples.append(
        generate_sample_dict(stitch_sample_group[0][Fields.start], stitch_sample_group[-1][Fields.end],
                             stitch_sample_group[0][Fields.label]))
    return processed_samples


def filter_view_labels(args, present_flags, samples):
    """
    Filters sequence samples based on "not-in-view" transitions in the present flags.
    If there is a sequence only part(s) of which is "not-in-view", removes that/those part(s) only, i.e. splits the
    original sequence into multiple subject-in-view sequences only
    :param args: program arguments
    :type present_flags: numpy.core.multiarray.ndarray
    :param present_flags: numpy array of booleans designating subject is present or not for each frame of the source video
    :type samples: list[OrderedDict]
    :param samples: unfiltered samples (prob. from multiview-format source)
    :return: list[OrderedDict]
    """
    samples_processed = []
    for sample in samples:
        ix_frame = sample[Fields.start]
        end_of_processing_range = sample[Fields.end] + 1
        sample_group = []
        while ix_frame < end_of_processing_range:
            # seek sample start
            while not present_flags[ix_frame] and ix_frame < end_of_processing_range:
                ix_frame += 1
            start_frame = ix_frame
            if ix_frame == end_of_processing_range:
                break
            while present_flags[ix_frame] and ix_frame < end_of_processing_range:
                ix_frame += 1
            end_frame = ix_frame - 1
            sample_group.append(generate_sample_dict(start_frame, end_frame, sample[Fields.label]))
        if len(sample_group) > 0:
            sample_group = stitch_samples(args, sample_group)
            length_filtered_group = [sample for sample in sample_group
                                     if sample[Fields.end] - sample[Fields.start] > args.min_sequence_length]
            samples_processed += length_filtered_group
    return samples_processed


def process_multiview_file(args, default_behavior_label):
    input_file_path = os.path.join(args.folder, args.input)
    file_handle = open(input_file_path, 'r')
    lines = tuple(file_handle)
    file_handle.close()
    header_line = lines[0]
    header_match_regex = re.compile(r"!(?:\s*\w+\s*,)+\s*\w+\s*")
    text_group_regex = re.compile(r"\w+")
    if header_match_regex.match(header_line) is None:
        raise ValueError("Incorrect format. The header line of the multiview-format label file should be: "
                         + "! view_name_1 [, view_name_2, view_name_3 ...]")
    view_names = text_group_regex.findall(header_line)

    samples, offset_points = parse_label_lines(args, lines[1:], multiview_mode=True)
    multiview_samples = []
    for _ in view_names:
        multiview_samples.append(duplicate_sample_list(samples))

    if not args.single_view_only:
        multiview_groups = []
        for ix_sample in range(len(samples)):
            group = []
            for samples, view in zip(multiview_samples, view_names):
                group.append([samples[ix_sample]])
            multiview_groups.append(group)

    multiview_samples_processed = []
    multiview_present_flags = []
    for view_name, view_samples in zip(view_names, multiview_samples):
        feature_file_name = view_name + "_features.npz"
        feature_path = os.path.join(args.folder, feature_file_name)
        if not os.path.isfile(feature_path):
            raise (IOError("Could not find feature file at {:s}. Each view in header of multiview-format label input " +
                           "file should have a corresponding feature file, <view name>_features_npz."
                           .format(feature_path)))
        present_flags = np.load(feature_path)['present']
        multiview_present_flags.append(present_flags)
        view_samples = filter_view_labels(args, present_flags, view_samples)
        view_samples = insert_out_of_frame_sequences(args, view_samples, present_flags=present_flags,
                                                     view_name=view_name)
        view_samples = stitch_samples(args, view_samples)

        if args.default_label:
            view_samples = insert_default_label_sequences(args, view_samples, default_behavior_label)

        multiview_samples_processed.append(view_samples)

    multiview_samples = multiview_samples_processed
    multiview_samples = process_multiview_offsets(multiview_samples, offset_points, view_names)

    if not args.multiview_only:
        for view_name, view_samples in zip(view_names, multiview_samples):
            # if given multiview-only setting, skip writing json label files for individual views

            if args.output_suffix:
                output_file = view_name + "_labels" + args.output_suffix + ".json"
            else:
                output_file = view_name + "_labels.json"
            output_path = os.path.join(args.folder, output_file)
            if args.verbosity > 0:
                print("Saving output to {:s}".format(output_file))
            file_handle = open(output_path, 'w')
            json.dump(view_samples, file_handle, indent=3)
            file_handle.close()

    if not args.single_view_only:
        processed_groups = []
        for group in multiview_groups:
            processed_group = []
            for sample_list, present_flags, view_name in zip(group, multiview_present_flags, view_names):
                sample_list = filter_view_labels(args, present_flags, sample_list)
                sample_list = insert_out_of_frame_sequences(args, sample_list, present_flags, view_name)
                sample_list = stitch_samples(args, sample_list)
                processed_group.append(sample_list)
            processed_groups.append(group)
        multiview_groups = processed_groups
        processed_groups = []

        if args.default_label:
            # insert default label groups between the currently-existing groups
            previous_group = [[generate_sample_dict(-1, -1, 0)] for _ in view_names]
            for current_group in multiview_groups:
                default_group = []
                for previous_list, current_list, present_flags, view_name in zip(previous_group, current_group,
                                                                                 multiview_present_flags, view_names):
                    default_list = [generate_sample_dict(previous_list[-1][Fields.end] + 1,
                                                         current_list[0][Fields.start] - 1,
                                                         default_behavior_label)]
                    default_list = filter_view_labels(args, present_flags, default_list)
                    default_list = insert_out_of_frame_sequences(args, default_list, present_flags, view_name)
                    default_list = stitch_samples(args, default_list)
                    default_group.append(default_list)
                processed_groups.append(default_group)
                processed_groups.append(current_group)
                previous_group = current_group
        multiview_groups = processed_groups
        processed_groups = []
        # remove "out-of-frame" & adjust offsets:
        for group in multiview_groups:
            processed_group = []
            empty_count = 0
            for sample_list in group:
                processed_list = []
                for sample in sample_list:
                    if sample[Fields.label] != label_mapping['X']:
                        processed_list.append(sample)
                if len(processed_list) == 0:
                    empty_count += 1
                processed_group.append(processed_list)
            processed_group = process_multiview_offsets(processed_group, offset_points, view_names)
            if empty_count < len(view_names):
                processed_groups.append(processed_group)
        multiview_groups = processed_groups
        output_and_print_multiview_groups(args, multiview_groups, view_names)


def main():
    args = process_arguments(Arguments, "text label -> JSON Label Converter for LSTM")

    default_behavior_label = 4

    if args.multiview_only and args.single_view_only:
        raise ValueError("{:s} and {:s} arguments cannot be combined.".format(Arguments.single_view_only.name,
                                                                              Arguments.multiview_only.name))

    if args.multiview_format:
        if args.input is None or (type(args.input) != str and len(args.input) != 1):
            raise ValueError("--multiview_format (-mf) mode requires the input argument to be set to the name of a " +
                             "single multiview-format label input file. Got: {:s}".format(str(args.input)))
        if type(args.input) != str and len(args.input) == 1:
            args.input = args.input[0]
        view_names = process_multiview_file(args, default_behavior_label)
    else:
        multiview_samples, view_names = process_individual_label_files(args, default_behavior_label)
        if args.single_view_only:
            return 0
        generate_multiview_labels(args, multiview_samples, view_names, default_behavior_label)
    return 0


if __name__ == "__main__":
    sys.exit(main())
