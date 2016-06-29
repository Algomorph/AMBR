#!/usr/bin/python3

import sys
import numpy as np
import os.path
import calib.io as cio
import calib.camera as cb
from calib.rig import Rig, MultiStereoRig
from calib.geom import Pose
import cv2
from enum import Enum
from multiprocessing import cpu_count


class Thresholds(Enum):
    MAX_SPATIAL_DIST_FROM_MEAN = 0.30


def load_frame_data(root_path, cams):
    data = []
    for cam in cams:
        data.append(np.load(root_path + "/" + cam + "/" + cam + "0_s_data.npz")['frame_data'])
    return data


def apply_extr(extr1, extr2):
    """
    :param extr1:
    :type extr1: calib.camera.Camera.Extrinsics
    :param extr2:
    :type extr2: calib.camera.Camera.Extrinsics
    :return:
    """
    pose1 = Pose(rotation=extr1.rotation, translation_vector=extr1.translation)
    pose2 = Pose(rotation=extr2.rotation, translation_vector=extr2.translation)
    pose3 = pose1.dot(pose2)
    return cb.Camera.Extrinsics(rotation=pose3.rmat, translation=pose3.tvec.reshape(1, 3))


def load_rig(root_path, cams):
    intrinsics = []

    for cam in cams:
        intrinsics.append(
            cio.load_opencv_calibration(os.path.join(root_path, "calib/intrinsics/" + cam + ".xml")).cameras[
                0].intrinsics)

    stereo_pair_labels = ["A", "E", "C", "D"]
    stereo_rigs = []
    stereo_extrinsics = []
    i_rig = 0

    for sp in stereo_pair_labels:
        stereo_rig = cio.load_opencv_calibration(os.path.join(root_path, "calib/stereo/" + sp + ".xml"))
        stereo_rig.label = stereo_pair_labels[i_rig]
        i_rig += 1
        stereo_rigs.append(stereo_rig)
        stereo_extrinsics.append(
            stereo_rig.cameras[1].extrinsics)

    cross_pairs = ["ar_el", "er_cl", "cr_dl", "dr_al"]
    cross_extrinsics = []
    for cp in cross_pairs:
        cross_extrinsics.append(
            cio.load_opencv_calibration(os.path.join(root_path, "calib/cross_pair/" + cp + ".xml")).cameras[
                1].extrinsics)

    cur_pose = cb.Camera.Extrinsics()
    global_extrinsics = [cur_pose]
    tvecs = [cur_pose.translation]

    for i_pair in range(len(stereo_pair_labels)):
        stereo_rigs[i_pair].extrinsics = cur_pose
        stereo_extr = stereo_extrinsics[i_pair]  # type: cb.Camera.Extrinsics
        cur_pose = apply_extr(cur_pose, stereo_extr)
        global_extrinsics.append(cur_pose)
        tvecs.append(cur_pose.translation)
        cross_extr = cross_extrinsics[i_pair]  # type: cb.Camera.Extrinsics
        cur_pose = apply_extr(cur_pose, cross_extr)
        global_extrinsics.append(cur_pose)
        tvecs.append(cur_pose.translation)

    global_extrinsics = global_extrinsics[:-1]
    tvecs = tvecs[:-1]
    tvecs = np.array(tvecs).reshape(8, 3)
    center = tvecs.mean(axis=0).reshape(1, 3)
    offset = -center

    # move center to center of cage
    for pose in global_extrinsics:
        pose.translation = pose.translation + offset
    print("Stereo camera positions (m):")
    for stereo_rig in stereo_rigs:
        print(stereo_rig.extrinsics.translation)

    cameras = []
    for i_cam in range(len(intrinsics)):
        cam = cb.Camera(intrinsics=intrinsics[i_cam], extrinsics=global_extrinsics[i_cam])
        cameras.append(cam)

    rig = Rig(cameras=cameras)

    ms_rig = MultiStereoRig(stereo_rigs=stereo_rigs)

    return rig, ms_rig


def compute_rectification(ms_rig):
    """
    :type ms_rig: MultiStereoRig
    :param ms_rig:
    :return:
    """
    for stereo_rig in ms_rig.rigs:
        im_size = stereo_rig.cameras[0].intrinsics.resolution
        rotation1, rotation2, pose1, pose2, Q = \
            cv2.stereoRectify(cameraMatrix1=stereo_rig.cameras[0].intrinsics.intrinsic_mat,
                              distCoeffs1=stereo_rig.cameras[0].intrinsics.distortion_coeffs,
                              cameraMatrix2=stereo_rig.cameras[1].intrinsics.intrinsic_mat,
                              distCoeffs2=stereo_rig.cameras[1].intrinsics.distortion_coeffs,
                              imageSize=(im_size[1], im_size[0]),
                              R=stereo_rig.cameras[1].extrinsics.rotation,
                              T=stereo_rig.cameras[1].extrinsics.translation,
                              flags=cv2.CALIB_ZERO_DISPARITY)[0:5]
        stereo_rig.cameras[0].stereo_rotation = rotation1
        stereo_rig.cameras[1].stereo_rotation = rotation2
        stereo_rig.cameras[0].stereo_pose = pose1
        stereo_rig.cameras[1].stereo_pose = pose2
        stereo_rig.extrinsics.inv_rotation = np.linalg.inv(stereo_rig.extrinsics.rotation)
        stereo_rig.Q = Q
        stereo_rig.f = Q[2, 3]
        stereo_rig.inv_baseline = Q[3, 2]
        stereo_rig.ox = Q[0, 3]
        stereo_rig.oy = Q[1, 3]


def coord_to_px(coord, px_per_m, out_c_horiz, out_c_vert):
    return (int(round(out_c_horiz + coord[0] * px_per_m)), int(round(out_c_vert - coord[1] * px_per_m)))


def orient_seg(coord, length, rmat):
    vec = np.array([[0], [0], [length]])
    v2 = rmat.dot(vec).reshape(1, 3)
    return coord + v2


def process_frames(frame_data, rig, ms_rig, root_path):
    cam_count = len(frame_data)
    prev_centers = [None] * cam_count
    more_frames_remain = True

    out_w = 1920
    out_h = 1080
    out_c_horiz = out_w // 2
    out_c_vert = out_h // 2

    writer = cv2.VideoWriter(os.path.join(root_path, "out.mpg"), cv2.VideoWriter_fourcc('X', '2', '6', '4'),
                             60, (out_w, out_h), True)
    writer.set(cv2.VIDEOWRITER_PROP_NSTRIPES, cpu_count())

    # ======== prepare template for output =====

    vis_pixels_per_m = 1000  # pixels per meter of physical space in the output video
    cage_side_m = .635
    cage_side_px = cage_side_m * vis_pixels_per_m
    cage_rect_vert1 = (int(round(out_c_horiz - cage_side_px / 2)), int(round(out_c_vert - cage_side_px / 2)))
    cage_rect_vert2 = (int(round(out_c_horiz + cage_side_px / 2)), int(round(out_c_vert + cage_side_px / 2)))
    out_template = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    out_template = cv2.rectangle(out_template, cage_rect_vert1, cage_rect_vert2, (255, 255, 255))
    v1 = cage_rect_vert1
    v2 = cage_rect_vert2

    # A, E, C, D
    marker_colors = [(255, 0, 0), (255, 204, 0), (0, 204, 0), (0, 204, 255)]
    line_vert_sets = np.array([(v1, (v1[0], v2[1])),
                               ((v1[0], v2[1]), v2),
                               ((v2[0], v1[1]), v2),
                               (v1, (v2[0], v1[1]))])
    line_centers = line_vert_sets.mean(axis=1)
    label_offsets = np.array([(-20, 0), (0, 22), (3, 0), (0, -8)])
    label_positions = (line_centers + label_offsets).astype(np.int32)
    i_rig = 0

    for stereo_rig in ms_rig.rigs:
        label = stereo_rig.label
        line_verts = line_vert_sets[i_rig]
        out_template = cv2.line(out_template, tuple(line_verts[0]), tuple(line_verts[1]), marker_colors[i_rig])
        transl0 = stereo_rig.extrinsics.translation
        rot0 = stereo_rig.extrinsics.rotation
        transl1 = rig.cameras[i_rig * 2 + 1].extrinsics.translation
        cam0_pos = coord_to_px((transl0[0, 2], transl0[0, 0]), vis_pixels_per_m, out_c_horiz, out_c_vert)
        arrow_tip = orient_seg(transl0, 0.05, rot0)
        cam0_dir = coord_to_px((arrow_tip[0, 2], arrow_tip[0, 0]), vis_pixels_per_m, out_c_horiz, out_c_vert)
        cv2.line(out_template, cam0_pos, cam0_dir, marker_colors[i_rig])
        cam1_pos = coord_to_px((transl1[0, 2], transl1[0, 0]), vis_pixels_per_m, out_c_horiz, out_c_vert)
        cv2.drawMarker(out_template, cam0_pos, marker_colors[i_rig], cv2.MARKER_CROSS, 20)
        cv2.drawMarker(out_template, cam1_pos, marker_colors[i_rig], cv2.MARKER_CROSS, 20)
        label_pos = label_positions[i_rig]
        cv2.putText(out_template, label, tuple(label_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, marker_colors[i_rig])
        i_rig += 1

    i_frame = 0
    while more_frames_remain and i_frame < 5000:
        # prepare center data:
        centers = []
        frame_number = int(frame_data[0][i_frame][0])
        for i_rig in range(len(ms_rig.rigs)):
            centroids = []
            stereo_rig = ms_rig.rigs[i_rig]
            for local_i_cam in range(2):
                i_cam = i_rig * 2 + local_i_cam
                datum = frame_data[i_cam][i_frame]
                if datum[1] == 0:
                    prev_centers[i_cam] = None
                    break
                centroid = np.array([datum[3], datum[4]])
                # filter out improperly-tracked centroid (too far from previous)
                if prev_centers[i_cam] is not None:
                    dist = np.linalg.norm(prev_centers[i_cam] - centroid)
                    if dist > 60:
                        break
                # undistort:
                intrinsics = rig.cameras[i_cam].intrinsics  # type: cb.Camera.Intrinsics
                pose = stereo_rig.cameras[local_i_cam].stereo_pose
                rotation = stereo_rig.cameras[local_i_cam].stereo_rotation
                centroid = cv2.undistortPoints(centroid.reshape((1, 1, 2)), intrinsics.intrinsic_mat,
                                               intrinsics.distortion_coeffs, R=rotation, P=pose)[0, 0]
                # if i_rig == 0:
                #     print(frame_number, local_i_cam, centroid)
                centroids.append(centroid)
            if len(centroids) == 2:
                disp = (centroids[0][0] - centroids[1][0])
                ib_d = stereo_rig.inv_baseline * disp
                z = stereo_rig.f / ib_d
                x = -(centroids[0][0] + stereo_rig.ox) / ib_d
                y = (centroids[0][1] + stereo_rig.oy) / ib_d

                center = np.array([[x, y, z]])
                # transform to global coordinate frame:
                center = center.dot(stereo_rig.extrinsics.inv_rotation) + stereo_rig.extrinsics.translation
                #print(stereo_rig.label, center - stereo_rig.extrinsics.translation)
                #center = stereo_rig.extrinsics.inv_rotation.dot((center + stereo_rig.extrinsics.translation).T)
                #center = (center + stereo_rig.extrinsics.translation).T
                centers.append((center.flatten(), i_rig))

        out_frame = out_template.copy()
        cv2.putText(out_frame, str(frame_number), (0, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 0))
        if len(centers) != 0:
            for center, i_rig in centers:
                # print(center)
                draw_pos = (out_c_horiz + int(center[2] * vis_pixels_per_m),
                            out_c_vert - int(center[0] * vis_pixels_per_m))
                # print(draw_pos)
                cv2.drawMarker(out_frame, draw_pos, marker_colors[i_rig], cv2.MARKER_CROSS, 20)
        writer.write(out_frame)
        i_frame += 1
    writer.release()


def main():
    cams = ['al', 'ar', 'el', 'er', 'cl', 'cr', 'dl', 'dr']
    root_path = "/media/algomorph/Data/reco/cap/mouse_cap04"
    frame_data = load_frame_data(root_path, cams)
    rig, ms_rig = load_rig(root_path, cams)
    compute_rectification(ms_rig)
    process_frames(frame_data, rig, ms_rig, root_path)


if __name__ == '__main__':
    sys.exit(main())
