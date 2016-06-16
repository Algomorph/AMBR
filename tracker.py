#!/usr/bin/python3

import sys
import numpy as np
import os.path
import calib.io as cio
import calib.camera as cb
from calib.rig import Rig, MultiStereoRig
from calib.geom import Pose
import cv2


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
    return cb.Camera.Extrinsics(rotation=pose3.rmat, translation=pose3.tvec)


def load_rig(root_path, cams):
    intrinsics = []

    for cam in cams:
        intrinsics.append(
            cio.load_opencv_calibration(os.path.join(root_path, "calib/intrinsics/" + cam + ".xml")).cameras[
                0].intrinsics)

    stereo_pairs = ["A", "C", "D", "E"]
    stereo_rigs = []
    stereo_extrinsics = []
    for sp in stereo_pairs:
        stereo_rig = cio.load_opencv_calibration(os.path.join(root_path, "calib/stereo/" + sp + ".xml"))
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
    tvecs = []
    for i_pair in range(len(stereo_pairs)):
        stereo_rigs[i_pair].extrinsics = cur_pose
        stereo_extr = stereo_extrinsics[i_pair]  # type: cb.Camera.Extrinsics
        cur_pose = apply_extr(cur_pose, stereo_extr)
        global_extrinsics.append(cur_pose)
        tvecs.append(cur_pose.translation.T)
        cross_extr = cross_extrinsics[i_pair]  # type: cb.Camera.Extrinsics
        cur_pose = apply_extr(cur_pose, cross_extr)
        global_extrinsics.append(cur_pose)
        tvecs.append(cur_pose.translation.T)

    global_extrinsics = global_extrinsics[:-1]
    tvecs = tvecs[:-1]
    tvecs = np.array(tvecs)
    center = tvecs.mean(axis=0).reshape(3, 1)
    offset = -center

    # move center to center of cage
    for pose in global_extrinsics:
        pose.translation = pose.translation + offset
    for stereo_rig in stereo_rigs:
        stereo_rig.extrinsics.translation = stereo_rig.extrinsics.translation + offset

    cameras = []
    for i_cam in range(len(intrinsics)):
        cam = cb.Camera(intrinsics=intrinsics[i_cam], extrinsics=global_extrinsics[i_cam])

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
        im_size = stereo_rig.cameras[0].resolution
        rotation1, rotation2, pose1, pose2 = \
            cv2.stereoRectify(cameraMatrix1=stereo_rig.cameras[0].intrinsics.intrinsic_mat,
                              distCoeffs1=stereo_rig.cameras[0].intrinsics.distortion_coeffs,
                              cameraMatrix2=stereo_rig.cameras[1].intrinsics.intrinsic_mat,
                              distCoeffs2=stereo_rig.cameras[1].intrinsics.distortion_coeffs,
                              imageSize=(im_size[1], im_size[0]),
                              R=stereo_rig.cameras[1].extrinsics.rotation,
                              T=stereo_rig.cameras[1].extrinsics.translation,
                              flags=cv2.CALIB_ZERO_DISPARITY
                              )[0:4]
        stereo_rig.cameras[0].stereo_rotation = rotation1
        stereo_rig.cameras[1].stereo_rotation = rotation2
        stereo_rig.cameras[0].stereo_pose = pose1
        stereo_rig.cameras[1].stereo_pose = pose2


def process_frames(frame_data, rig, ms_rig):
    cam_count = len(frame_data)
    prev_centers = [None] * cam_count
    more_frames_remain = True
    i_frame = 0
    while more_frames_remain:
        # prepare center data:
        centers = []
        for i_rig in range(ms_rig.rigs):
            centroids = []
            stereo_rig = ms_rig[i_rig]
            for local_i_cam in range(2):
                i_cam = i_rig * 2 + local_i_cam
                datum = frame_data[i_cam][i_frame]
                if datum[1] == 0:
                    prev_centers[i_cam] = None
                    break
                centroid = [datum[3], datum[4]]
                # filter out improperly-tracked centroid (too far from previous)
                if prev_centers[i_cam] is not None:

                    dist = np.linalg.norm(prev_centers[i_cam] - centroid)
                    if dist > 60:
                        break
                # undistort:
                intrinsics = rig.cameras[i_cam].intrinsics  # type: cb.Camera.Intrinsics
                pose = stereo_rig.cameras[local_i_cam].stereo_pose
                rotation = stereo_rig.cameras[local_i_cam].stereo_rotation
                centroid = cv2.undistortPoints([[centroid]], intrinsics.intrinsic_mat, intrinsics.distortion_coeffs,
                                               R=rotation, P=pose)[0, 0]
                centroids.append(centroid)
            if len(centroids) == 2:
                baseline = stereo_rig.cameras[1].stereo_pose[0, 3]
                x = centroids[0][0]
                y = centroids[0][1]
                z = baseline / (centroids[0][0] - centroids[1][0])
                center = np.array([[x, y, z]])
                center.dot(stereo_rig.extrinsics.rotation) + stereo_rig.extrinsics.translation
                # transform to global coordinate frame:

        if len(centers) == 0:
            continue

        # TODO: average and output them centers


def main():
    cams = ['al', 'ar', 'cl', 'cr', 'el', 'er', 'dl', 'dr']
    root_path = "/media/algomorph/Data/reco/cap/mouse_cap04"
    frame_data = load_frame_data(root_path, cams)
    rig, ms_rig = load_rig(root_path, cams)
    compute_rectification(ms_rig)
    process_frames(frame_data, rig, ms_rig)


if __name__ == '__main__':
    sys.exit(main())
