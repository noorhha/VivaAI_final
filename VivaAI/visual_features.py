"""
visual_features.py
-------------------
Visual feature extraction using MediaPipe face landmarker.

Ported exactly from 01_VivaAI_Feature_Extraction.ipynb (cell 11) so that
live predictions use features computed identically to training data.

Produces, per 3-second window:
    vis_face_detected_ratio, vis_off_camera_ratio, vis_blink_frequency,
    vis_head_motion_mean, vis_head_pose_variance, vis_gaze_stability,
    vis_gaze_offcenter_ratio, vis_head_down_ratio, vis_facial_activity
"""

from collections import deque, defaultdict
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    RunningMode,
)

# Landmark indices - must match training exactly
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

VISUAL_SAMPLE_FPS = 15.0

MIN_VIDEO_BRIGHTNESS = 25.0
MAX_VIDEO_BRIGHTNESS = 235.0
MIN_VIDEO_BLUR_SCORE = 15.0

FACE_MODEL_PATH = Path(__file__).parent / "face_landmarker.task"


def ear_score(points):
    """Eye aspect ratio for a set of 6 eye landmark points."""
    a = np.linalg.norm(points[1] - points[5])
    b = np.linalg.norm(points[2] - points[4])
    c = np.linalg.norm(points[0] - points[3])
    return float((a + b) / (2.0 * c + 1e-8))


def nan_mean(values, default=0.0):
    values = np.asarray(values, dtype=float)
    valid = values[np.isfinite(values)]
    return float(np.mean(valid)) if len(valid) else float(default)


def nan_std(values, default=0.0):
    values = np.asarray(values, dtype=float)
    valid = values[np.isfinite(values)]
    return float(np.std(valid)) if len(valid) else float(default)


def extract_visual_timeseries(video_path, face_model_path=FACE_MODEL_PATH):
    """
    Runs MediaPipe face landmarker across sampled frames of the video.

    Returns:
        data       -- dict of per-sampled-frame arrays (t, face_detected,
                       blink_event, yaw, pitch, gaze_x, gaze_y,
                       facial_activity, brightness, blur_score)
        metadata   -- dict of whole-video summary stats (brightness,
                       blur, sample_fps, sampled_frames, width, height)
    """
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError("OpenCV could not open the video.")

    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    frame_step = max(1, int(round(source_fps / VISUAL_SAMPLE_FPS)))

    options = FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(face_model_path)),
        running_mode=RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=True,
    )

    values = defaultdict(list)
    existing_landmarks = None
    eye_closed = False
    ear_history = deque(maxlen=90)
    frame_index = 0
    sampled_frames = 0

    with FaceLandmarker.create_from_options(options) as detector:
        while True:
            success, frame = cap.read()
            if not success:
                break

            if frame_index % frame_step != 0:
                frame_index += 1
                continue

            timestamp = frame_index / source_fps
            sampled_frames += 1

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = float(np.mean(gray))
            blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

            values["brightness"].append(brightness)
            values["blur_score"].append(blur_score)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = detector.detect(image)

            values["t"].append(timestamp)

            if not result.face_landmarks:
                values["face_detected"].append(0.0)
                values["blink_event"].append(0.0)
                values["yaw"].append(np.nan)
                values["pitch"].append(np.nan)
                values["gaze_x"].append(np.nan)
                values["gaze_y"].append(np.nan)
                values["facial_activity"].append(np.nan)

                existing_landmarks = None
                eye_closed = False

                frame_index += 1
                continue

            landmarks = result.face_landmarks[0]

            def point(index):
                return np.array(
                    [landmarks[index].x * width, landmarks[index].y * height],
                    dtype=float,
                )

            values["face_detected"].append(1.0)

            left_eye_points = np.array([point(i) for i in LEFT_EYE])
            right_eye_points = np.array([point(i) for i in RIGHT_EYE])

            ear = (ear_score(left_eye_points) + ear_score(right_eye_points)) / 2.0
            ear_history.append(ear)

            baseline_ear = (
                float(np.percentile(ear_history, 75))
                if len(ear_history) >= 10
                else 0.25
            )

            currently_closed = ear < baseline_ear * 0.75
            blink_event = 0.0

            if currently_closed and not eye_closed:
                blink_event = 1.0
                eye_closed = True
            elif not currently_closed:
                eye_closed = False

            values["blink_event"].append(blink_event)

            yaw = np.nan
            pitch = np.nan

            if result.facial_transformation_matrixes:
                matrix = np.asarray(
                    result.facial_transformation_matrixes[0], dtype=float
                )
                rotation = matrix[:3, :3]

                pitch = float(
                    np.degrees(
                        np.arctan2(
                            -rotation[2, 0],
                            np.sqrt(rotation[2, 1] ** 2 + rotation[2, 2] ** 2),
                        )
                    )
                )

                yaw = float(
                    np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0]))
                )

            values["yaw"].append(yaw)
            values["pitch"].append(pitch)

            left_iris = np.mean(
                [[landmarks[i].x, landmarks[i].y] for i in LEFT_IRIS], axis=0
            )
            right_iris = np.mean(
                [[landmarks[i].x, landmarks[i].y] for i in RIGHT_IRIS], axis=0
            )

            left_eye_xy = np.array(
                [[landmarks[i].x, landmarks[i].y] for i in LEFT_EYE], dtype=float
            )
            right_eye_xy = np.array(
                [[landmarks[i].x, landmarks[i].y] for i in RIGHT_EYE], dtype=float
            )

            left_min = left_eye_xy.min(axis=0)
            left_span = np.maximum(left_eye_xy.max(axis=0) - left_min, 1e-6)

            right_min = right_eye_xy.min(axis=0)
            right_span = np.maximum(right_eye_xy.max(axis=0) - right_min, 1e-6)

            left_gaze = (left_iris - left_min) / left_span
            right_gaze = (right_iris - right_min) / right_span
            normalized_gaze = (left_gaze + right_gaze) / 2.0

            values["gaze_x"].append(float(np.clip(normalized_gaze[0], 0.0, 1.0)))
            values["gaze_y"].append(float(np.clip(normalized_gaze[1], 0.0, 1.0)))

            current_landmarks = np.array(
                [[lm.x, lm.y] for lm in landmarks], dtype=float
            )

            inter_eye_distance = float(
                np.linalg.norm(
                    np.mean(left_eye_points, axis=0)
                    - np.mean(right_eye_points, axis=0)
                )
            )

            if existing_landmarks is None:
                activity = 0.0
            else:
                raw_activity = float(
                    np.linalg.norm(
                        current_landmarks - existing_landmarks, axis=1
                    ).mean()
                )
                activity = raw_activity / max(
                    inter_eye_distance / max(width, 1), 1e-6
                )

            values["facial_activity"].append(activity)
            existing_landmarks = current_landmarks
            frame_index += 1

    cap.release()

    data = {key: np.asarray(value, dtype=float) for key, value in values.items()}

    brightness_values = data.get("brightness", np.asarray([], dtype=float))
    blur_values = data.get("blur_score", np.asarray([], dtype=float))

    metadata = {
        "source_fps": source_fps,
        "sample_fps": source_fps / frame_step,
        "sampled_frames": sampled_frames,
        "width": width,
        "height": height,
        "video_brightness_mean": (
            float(np.mean(brightness_values)) if len(brightness_values) else np.nan
        ),
        "video_brightness_std": (
            float(np.std(brightness_values)) if len(brightness_values) else np.nan
        ),
        "video_dark_frame_ratio": (
            float(np.mean(brightness_values < MIN_VIDEO_BRIGHTNESS))
            if len(brightness_values)
            else np.nan
        ),
        "video_bright_frame_ratio": (
            float(np.mean(brightness_values > MAX_VIDEO_BRIGHTNESS))
            if len(brightness_values)
            else np.nan
        ),
        "video_blur_score": (
            float(np.mean(blur_values)) if len(blur_values) else np.nan
        ),
        "video_blurry_frame_ratio": (
            float(np.mean(blur_values < MIN_VIDEO_BLUR_SCORE))
            if len(blur_values)
            else np.nan
        ),
    }

    return data, metadata


def visual_features_for_window(visual_data, start_time, end_time):
    """Aggregates the per-frame visual timeseries into one window's features."""
    timestamps = visual_data["t"]
    mask = (timestamps >= start_time) & (timestamps < end_time)

    window_duration = max(end_time - start_time, 1e-6)

    if not np.any(mask):
        return {
            "vis_face_detected_ratio": 0.0,
            "vis_off_camera_ratio": 1.0,
            "vis_blink_frequency": 0.0,
            "vis_head_motion_mean": 0.0,
            "vis_head_pose_variance": 0.0,
            "vis_gaze_stability": 0.0,
            "vis_gaze_offcenter_ratio": 1.0,
            "vis_head_down_ratio": 0.0,
            "vis_facial_activity": 0.0,
        }

    face = visual_data["face_detected"][mask]
    yaw = visual_data["yaw"][mask]
    pitch = visual_data["pitch"][mask]
    gaze_x = visual_data["gaze_x"][mask]
    gaze_y = visual_data["gaze_y"][mask]
    activity = visual_data["facial_activity"][mask]
    blink_events = visual_data["blink_event"][mask]

    face_ratio = float(np.mean(face))
    off_camera_ratio = 1.0 - face_ratio

    valid_yaw = yaw[np.isfinite(yaw)]
    valid_pitch = pitch[np.isfinite(pitch)]

    yaw_diff = np.abs(np.diff(valid_yaw)) if len(valid_yaw) >= 2 else np.array([])
    pitch_diff = (
        np.abs(np.diff(valid_pitch)) if len(valid_pitch) >= 2 else np.array([])
    )

    head_motion_values = np.concatenate([yaw_diff, pitch_diff])
    head_motion_mean = (
        float(np.mean(head_motion_values)) if len(head_motion_values) else 0.0
    )

    head_pose_variance = nan_std(valid_yaw) ** 2 + nan_std(valid_pitch) ** 2

    gaze_pairs = np.column_stack([gaze_x, gaze_y])
    valid_gaze_mask = np.isfinite(gaze_pairs).all(axis=1)
    valid_gaze = gaze_pairs[valid_gaze_mask]

    if len(valid_gaze) >= 2:
        gaze_variance = float(np.var(valid_gaze, axis=0).sum())
        gaze_stability = 1.0 / (1.0 + gaze_variance * 100.0)

        offcenter = (
            (valid_gaze[:, 0] < 0.30)
            | (valid_gaze[:, 0] > 0.70)
            | (valid_gaze[:, 1] < 0.20)
            | (valid_gaze[:, 1] > 0.80)
        )
        gaze_offcenter_ratio = float(np.mean(offcenter))
    else:
        gaze_stability = 0.0
        gaze_offcenter_ratio = 1.0

    all_valid_pitch = visual_data["pitch"][np.isfinite(visual_data["pitch"])]
    recording_pitch_median = (
        float(np.median(all_valid_pitch)) if len(all_valid_pitch) else 0.0
    )

    head_down_ratio = (
        float(np.mean(valid_pitch > recording_pitch_median + 10.0))
        if len(valid_pitch)
        else 0.0
    )

    return {
        "vis_face_detected_ratio": round(face_ratio, 5),
        "vis_off_camera_ratio": round(off_camera_ratio, 5),
        "vis_blink_frequency": round(
            float(np.sum(blink_events)) / window_duration, 5
        ),
        "vis_head_motion_mean": round(head_motion_mean, 5),
        "vis_head_pose_variance": round(head_pose_variance, 5),
        "vis_gaze_stability": round(gaze_stability, 5),
        "vis_gaze_offcenter_ratio": round(gaze_offcenter_ratio, 5),
        "vis_head_down_ratio": round(head_down_ratio, 5),
        "vis_facial_activity": round(nan_mean(activity), 6),
    }
