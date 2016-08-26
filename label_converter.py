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


def generate_sample_dict(beginning_frame, end_frame, label):
    sample = OrderedDict()
    sample[Fields.start] = int(beginning_frame)
    sample[Fields.end] = int(end_frame)
    sample[Fields.label] = int(label)
    return sample


def sample_end(sample):
    if type(sample) == list:
        return sample[-1][Fields.end]
    return sample[Fields.end]


def sample_len(sample):
    if type(sample) == list:
        return sample[-1][Fields.end] - sample[0][Fields.start] + 1
    return sample[Fields.end] - sample[Fields.start] + 1


def samples_match_check(samples, offsets, view_names, timestep_threshold):
    ix_source_view = 0
    source_view = None
    source_sample = None
    seek_offsets = [0] * len(view_names)

    max_length = 0

    # find longest sample to check against
    for ix_view, sample in enumerate(samples):
        sample_length = sample_len(sample)
        if sample[Fields.label] != 0 and sample_length > max_length:
            max_length = sample_length
            source_sample = sample
            source_view = view_names[ix_view]
            ix_source_view = ix_view

    if source_sample is None:
        return False, ix_source_view, seek_offsets

    source_beginning = source_sample[Fields.start] + offsets[ix_source_view]
    source_end = source_sample[Fields.end] + offsets[ix_source_view]
    source_offset = offsets[ix_source_view]
    source_label = source_sample[Fields.label]

    for ix_view, target_sample in enumerate(samples):
        if ix_view != ix_source_view:
            target_view = view_names[ix_view]
            target_offset = offsets[ix_view]

            if target_sample[Fields.label] == 0:
                seek_offsets[ix_view] = target_sample[Fields.end] + 1
                continue

            target_beginning = target_sample[Fields.start] + offsets[ix_view]
            target_end = target_sample[Fields.end] + offsets[ix_view]
            target_label = target_sample[Fields.label]
            if source_label != target_label:
                raise ValueError("Mismatch between sample {:s}({:s}) and {:s}({:s}) labels."
                                 .format(str(source_sample), source_view, str(target_sample), target_view))
            if not (abs(source_beginning - target_beginning) < timestep_threshold):
                raise ValueError("Mismatch between sample {:s}({:s} + {:d}) and {:s}({:s} + {:d}) start times."
                                 .format(str(source_sample), source_view, source_offset, str(target_sample),
                                         target_view, target_offset))

            if not (abs(source_end - target_end) < timestep_threshold):
                seek_offsets[ix_view] = target_sample[Fields.end] + 1
                offsets[ix_view] += source_beginning - target_beginning // 1
            else:
                offsets[ix_view] += int(round(((source_beginning - target_beginning) + (source_end - target_end)) / 2))

    return True, ix_source_view, seek_offsets


def seek_within_samples(sample_streams, view_names, ix_source_view, active_samples, seek_offsets, offsets,
                        timestep_threshold):
    if np.sum(seek_offsets) == 0:
        return active_samples
    source_sample = active_samples[ix_source_view]
    source_view = view_names[ix_source_view]
    source_offset = offsets[ix_source_view]
    source_label = source_sample[Fields.label]
    for ix_view, stream in enumerate(sample_streams):
        if seek_offsets[ix_view] == 0:
            continue
        target_samples = []
        target_view = view_names[ix_view]
        target_offset = offsets[ix_view]
        current_label = active_samples[ix_view][Fields.label]
        have_match = current_label == source_label
        if have_match:
            target_samples.append(active_samples[ix_view])
        target_stream = sample_streams[ix_view]
        for ix_frame in range(seek_offsets[ix_view], source_sample[Fields.end] + timestep_threshold):
            if ix_frame in target_stream:
                target_sample = target_stream[ix_frame]
                if target_sample[Fields.label] == 0:
                    ix_frame += target_sample[Fields.end] - target_sample[Fields.start] + 1
                    continue
                if target_sample[Fields.label] == source_label:
                    have_match = True
                    if target_sample[Fields.end] + target_offset > source_sample[
                        Fields.end] + source_offset + timestep_threshold:
                        raise ValueError("Mismatch between sample {:s}({:s} + {:d}) and {:s}({:s} + {:d}) end times."
                                         .format(str(source_sample), source_view, source_offset, str(target_sample),
                                                 target_view, target_offset))
                    target_samples.append(target_sample)
                else:
                    if abs(source_sample[Fields.end] + source_offset - target_sample[
                        Fields.start] + target_offset) > timestep_threshold:
                        raise ValueError("Mismatch between sample {:s}({:s}) and {:s}({:s}) labels."
                                         .format(str(source_sample), source_view, str(target_sample), target_view))
        if have_match:
            if len(target_samples) > 1:
                active_samples[ix_view] = target_samples
            else:
                active_samples[ix_view] = target_samples[0]

    return active_samples


def clear_path_if_present(path):
    if os.path.exists(path):
        if os.path.isfile(path):
            os.remove(path)
        else:
            shutil.rmtree(path)


def label_at_time(sample, time):
    if type(sample) != list:
        return sample[Fields.label]
    else:
        ix = 0
        while ix < len(sample) and sample[ix][Fields.end] > time:
            ix += 1
        return sample[ix - 1][Fields.label]


def print_matched_set(matched_set, print_endpoints=False):
    start = sys.maxsize
    end = 0
    max_len = 1
    for sample in matched_set:
        if type(sample) == list:
            cur_start = sample[0][Fields.start]
            cur_end = sample[-1][Fields.end]
            if len(sample) > max_len:
                max_len = len(sample)
        else:
            cur_start = sample[Fields.start]
            cur_end = sample[Fields.end]
        if cur_start < start:
            start = cur_start
        if cur_end > end:
            end = cur_end
    label_array = np.zeros((max_len, len(matched_set)), dtype=np.int32)
    border_array = np.zeros((max_len - 1, len(matched_set)), dtype=np.bool)
    time_increment = end - start / max_len
    current_time = start
    current_labels = np.zeros(shape=(len(matched_set),), dtype=np.int32)
    for ix_sample, sample in enumerate(matched_set):
        current_labels[ix_sample] = label_at_time(sample, current_time)
    label_array[0] = current_labels
    previous_labels = current_labels
    current_time += time_increment
    for ix_step in range(1, max_len):
        for ix_sample, sample in enumerate(matched_set):
            current_labels[ix_sample] = label_at_time(sample, current_time)
            if current_labels[ix_sample] != previous_labels[ix_sample]:
                border_array[ix_step - 1, ix_sample] = True
            label_array[ix_step] = current_labels
        current_time += time_increment

    for ix_sample in range(0, len(matched_set)):
        print("|              |", end="")
    print("\n", end="")
    for ix_step in range(0, max_len - 1):
        for ix_sample in range(0, len(matched_set)):
            print("|{:^14}|".format(label_array[ix_step, ix_sample]), end="")
        print("\n", end="")
        for ix_sample in range(0, len(matched_set)):
            if border_array[ix_step, ix_sample]:
                print("================", end="")
            else:
                print("|              |", end="")
        print("\n", end="")
    for ix_sample in range(0, len(matched_set)):
        print("|{:^14}|".format(label_array[-1, ix_sample]), end="")
    print("\n", end="")
    for ix_sample in range(0, len(matched_set)):
        print("|              |", end="")
    print("\n", end="")
    for ix_sample in range(0, len(matched_set)):
        print("================", end="")
    print("\n", end="")


def print_multiview_headers(view_names):
    width = len(view_names)
    for i in view_names:
        print("================", end="")
    sys.stdout.write("\n")
    for name in view_names:
        print("|              |", end="")
    print("\n", end="")
    for name in view_names:
        print("|{:^14}|".format(name), end="")
    print("\n", end="")
    for name in view_names:
        print("|              |", end="")
    print("\n", end="")
    for i in view_names:
        print("================", end="")
    print("\n", end="")


def print_matched_labels(matched_labels, view_names):
    print_multiview_headers(view_names)
    for set in matched_labels:
        print_matched_set(set)


def main():
    args = process_arguments(Arguments, "text label -> JSON Label Converter for LSTM")
    verbosity = 1
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
        all_files_in_dir.sort()
        for file_name in all_files_in_dir:
            file_path = os.path.join(args.folder, file_name)
            if os.path.isfile(file_path) and file_name.endswith(default_input_file_ending):
                args.input.append(file_name)
                view_name = file_name[:-len(default_input_file_ending)]
                feature_file_name = view_name + "_features.npz"
                feature_path = os.path.join(args.folder, feature_file_name)
                if not os.path.isfile(feature_path):
                    raise (IOError("Cound not find feature file at {:s}. Each label text file should have a " +
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

    if args.single_view_only:
        return 0

    multiview_samples = [{sample[Fields.start]: sample for sample in multiview_sample_set} for multiview_sample_set in
                         multiview_samples]

    matched_samples = []
    samples_per_matched_set = len(view_names)
    offsets = [0] * samples_per_matched_set
    active_samples = [None] * samples_per_matched_set
    samples_for_matching_count = 0

    ix_frame = 0
    print_multiview_headers(view_names)
    end_at = sys.maxsize
    while ix_frame < frame_count:
        for ix_view, view in enumerate(view_names):
            if active_samples[ix_view] is None and ix_frame in multiview_samples[ix_view]:
                sample = multiview_samples[ix_view][ix_frame]
                active_samples[ix_view] = sample
                new_end_at = sample[Fields.end]
                if new_end_at < end_at:
                    end_at = new_end_at
                samples_for_matching_count += 1

        if samples_for_matching_count == samples_per_matched_set or ix_frame == end_at:
            retval, ix_source_view, seek_offsets = samples_match_check(active_samples, offsets, view_names,
                                                                       args.time_offset_threshold)
            if retval:
                active_samples = seek_within_samples(multiview_samples, view_names, ix_source_view,
                                                     active_samples, seek_offsets, offsets, args.time_offset_threshold)
                matched_samples.append(active_samples)
            print_matched_set(active_samples)
            advance_to = np.min([sample_end(sample) + 1 for sample in active_samples]) - args.time_offset_threshold
            # reset matched set
            active_samples = [None] * samples_per_matched_set
            ix_frame = advance_to
            samples_for_matching_count = 0
            end_at = sys.maxsize
        else:
            ix_frame += 1

    output_path = os.path.join(args.folder, "multiview_samples.json")
    file_handle = open(output_path, 'w')
    json.dump(matched_samples, file_handle, indent=3)
    file_handle.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
