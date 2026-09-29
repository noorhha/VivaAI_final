"""
audio_features.py
------------------
Audio feature extraction using ffmpeg + Librosa.

Ported exactly from 01_VivaAI_Feature_Extraction.ipynb (cell 12) so that
live predictions use features computed identically to training data.

Produces, per 3-second window:
    aud_pitch_mean, aud_pitch_std, aud_pitch_range, aud_pitch_slope,
    aud_energy_mean, aud_energy_std, aud_energy_slope, aud_voice_jitter,
    aud_shimmer, aud_pause_ratio, aud_pause_count, aud_mean_pause_duration,
    aud_total_pause_duration, aud_voiced_ratio, aud_zcr_mean

Also produces whole-answer audio quality stats used for quality flags:
    audio_duration_sec, audio_rms_mean, audio_rms_std,
    audio_silence_threshold_db, audio_clip_ratio, audio_low_volume_ratio,
    audio_silence_ratio, audio_snr_proxy_db, audio_mean_voiced_ratio
"""

import math
import subprocess

import numpy as np
import librosa

WINDOW_SEC = 3.0
STEP_SEC = 3.0  # non-overlapping, matches training (USE_OVERLAP=False)

# Training answers (video_duration_sec in answers.csv) had a median of
# ~12.5s and 75th percentile of ~15s; live exam answers can run up to 30s.
# That's well outside the training distribution, so a 30s answer produces
# roughly 10 windows to aggregate into __mean/__std/__slope features vs.
# the ~4-5 windows the model actually learned from, so the aggregated
# statistics behave differently even for genuinely authentic behavior.
# Capping window generation at the 75th percentile keeps the vast majority
# of the training distribution intact while closing most of the gap with
# the 30s live limit. This only affects how many WINDOWS get generated.
# whole-answer quality stats (audio_duration_sec, silence_ratio, etc.)
# below still reflect the true full-length recording, since those aren't
# part of what the GRU/RF were trained on and quality flags should see
# the real audio.
TRAINING_MATCH_DURATION_SEC = 15.0


def safe_linear_slope(values):
    """Least-squares slope of a 1D series. Returns 0 if too few/constant points."""
    values = np.asarray(values, dtype=float)

    if len(values) < 3 or np.allclose(values, values[0]):
        return 0.0

    x = np.arange(len(values), dtype=float)

    try:
        return float(np.polyfit(x, values, 1)[0])
    except Exception:
        return 0.0


def contiguous_true_runs(mask):
    """Returns list of (start_index, end_index) for contiguous True runs in a boolean mask."""
    runs = []
    start = None

    for index, value in enumerate(mask):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append((start, index))
            start = None

    if start is not None:
        runs.append((start, len(mask)))

    return runs


def extract_audio_windows(video_path, temp_wav_path):
    """
    Extracts audio from video via ffmpeg, computes pitch/energy/pause
    features per 3-second window using Librosa.

    Args:
        video_path:     path to the source video file
        temp_wav_path:  path to write/delete a temporary mono 16kHz wav

    Returns:
        windows        -- list of dicts, one per window
        answer_quality -- dict of whole-answer audio quality stats
    """
    command = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", "16000",
        str(temp_wav_path),
    ]

    result = subprocess.run(command, capture_output=True, text=True)

    # A non-zero exit with empty stderr usually means a transient file-access
    # conflict (e.g. the same source file was just read by Whisper's own
    # ffmpeg call moments earlier) rather than a genuinely corrupted video.
    # One short retry resolves this in practice without masking real errors.
    # If ffmpeg reports an actual message in stderr, that's a real failure
    # and retrying won't help, so we only retry the silent-failure case.
    if result.returncode != 0 and not result.stderr.strip():
        import time
        time.sleep(2)
        result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode != 0 or not temp_wav_path.exists():
        raise RuntimeError(f"FFmpeg audio extraction failed: {result.stderr}")

    try:
        audio, sample_rate = librosa.load(str(temp_wav_path), sr=16000, mono=True)
    finally:
        if temp_wav_path.exists():
            temp_wav_path.unlink()

    if len(audio) == 0:
        raise RuntimeError("Extracted audio is empty.")

    duration = len(audio) / sample_rate
    hop_length = 512
    frame_rate = sample_rate / hop_length

    rms = librosa.feature.rms(y=audio, hop_length=hop_length)[0]
    rms_db = librosa.amplitude_to_db(rms + 1e-10, ref=np.max)
    zcr = librosa.feature.zero_crossing_rate(audio, hop_length=hop_length)[0]

    f0, voiced_flag, voiced_probability = librosa.pyin(
        audio, fmin=50, fmax=500, sr=sample_rate, hop_length=hop_length
    )
    f0 = np.asarray(f0, dtype=float)
    voiced_flag = np.asarray(voiced_flag, dtype=bool)

    noise_floor = float(np.percentile(rms_db, 20))
    silence_threshold_db = min(-25.0, noise_floor + 6.0)

    clip_ratio = float(np.mean(np.abs(audio) >= 0.99))
    low_volume_ratio = float(np.mean(rms < 0.01))
    silence_ratio = float(np.mean(rms_db < silence_threshold_db))
    snr_proxy_db = float(
        np.percentile(rms_db, 90) - np.percentile(rms_db, 10)
    )

    global_voiced_mask = voiced_flag & np.isfinite(f0) & (f0 > 0)
    global_voiced_ratio = (
        float(np.mean(global_voiced_mask)) if len(global_voiced_mask) else 0.0
    )

    windows = []
    start_time = 0.0
    window_index = 0

    # Cap window generation to match training's answer-length distribution
    # (see TRAINING_MATCH_DURATION_SEC above). answer_quality below still
    # uses the true, uncapped `duration` for quality flags.
    windowing_duration = min(duration, TRAINING_MATCH_DURATION_SEC)

    while start_time < windowing_duration:
        end_time = min(start_time + WINDOW_SEC, windowing_duration)

        frame_start = int(start_time * frame_rate)
        frame_end = min(int(math.ceil(end_time * frame_rate)), len(rms))

        rms_window = rms[frame_start:frame_end]
        rms_db_window = rms_db[frame_start:frame_end]
        zcr_window = zcr[frame_start:frame_end]
        f0_window = f0[frame_start:frame_end]
        voiced_flag_window = voiced_flag[frame_start:frame_end]

        voiced_mask = voiced_flag_window & np.isfinite(f0_window) & (f0_window > 0)
        valid_pitch = f0_window[voiced_mask]

        pitch_mean = float(np.mean(valid_pitch)) if len(valid_pitch) else 0.0
        pitch_std = float(np.std(valid_pitch)) if len(valid_pitch) >= 2 else 0.0
        pitch_range = float(np.ptp(valid_pitch)) if len(valid_pitch) >= 2 else 0.0

        pitch_slope = (
            safe_linear_slope(valid_pitch) / max(pitch_mean, 1.0)
            if len(valid_pitch) >= 3
            else 0.0
        )

        jitter_proxy = (
            float(
                np.mean(np.abs(np.diff(valid_pitch)))
                / max(np.mean(valid_pitch), 1e-6)
            )
            if len(valid_pitch) >= 3
            else 0.0
        )

        voiced_ratio = float(np.mean(voiced_mask)) if len(voiced_mask) else 0.0

        energy_mean = float(np.mean(rms_window)) if len(rms_window) else 0.0
        energy_std = float(np.std(rms_window)) if len(rms_window) else 0.0

        energy_slope = (
            safe_linear_slope(rms_window) / max(energy_mean, 1e-8)
            if len(rms_window) >= 3
            else 0.0
        )

        voiced_energy = rms_window[voiced_mask]
        shimmer_proxy = (
            float(
                np.mean(np.abs(np.diff(voiced_energy)))
                / max(np.mean(voiced_energy), 1e-8)
            )
            if len(voiced_energy) >= 3
            else 0.0
        )

        silence_mask = rms_db_window < silence_threshold_db
        pause_durations = []

        for run_start, run_end in contiguous_true_runs(silence_mask):
            pause_duration = (run_end - run_start) / frame_rate
            if pause_duration >= 0.20:
                pause_durations.append(pause_duration)

        current_duration = max(end_time - start_time, 1e-6)
        total_pause_duration = float(sum(pause_durations))
        pause_ratio = min(1.0, total_pause_duration / current_duration)

        windows.append({
            "window_idx": window_index,
            "win_start": round(start_time, 3),
            "win_end": round(end_time, 3),
            "aud_pitch_mean": round(pitch_mean, 5),
            "aud_pitch_std": round(pitch_std, 5),
            "aud_pitch_range": round(pitch_range, 5),
            "aud_pitch_slope": round(pitch_slope, 6),
            "aud_energy_mean": round(energy_mean, 8),
            "aud_energy_std": round(energy_std, 8),
            "aud_energy_slope": round(energy_slope, 6),
            "aud_voice_jitter": round(jitter_proxy, 6),
            "aud_shimmer": round(shimmer_proxy, 6),
            "aud_pause_ratio": round(pause_ratio, 6),
            "aud_pause_count": int(len(pause_durations)),
            "aud_mean_pause_duration": round(
                float(np.mean(pause_durations)) if pause_durations else 0.0, 6
            ),
            "aud_total_pause_duration": round(total_pause_duration, 6),
            "aud_voiced_ratio": round(voiced_ratio, 6),
            "aud_zcr_mean": round(
                float(np.mean(zcr_window)) if len(zcr_window) else 0.0, 6
            ),
        })

        start_time += STEP_SEC
        window_index += 1

    answer_quality = {
        "audio_duration_sec": round(duration, 4),
        "audio_rms_mean": round(float(np.mean(rms)), 8),
        "audio_rms_std": round(float(np.std(rms)), 8),
        "audio_silence_threshold_db": round(silence_threshold_db, 4),
        "audio_clip_ratio": round(clip_ratio, 8),
        "audio_low_volume_ratio": round(low_volume_ratio, 6),
        "audio_silence_ratio": round(silence_ratio, 6),
        "audio_snr_proxy_db": round(snr_proxy_db, 4),
        "audio_mean_voiced_ratio": round(global_voiced_ratio, 6),
    }

    return windows, answer_quality
