"""
feature_extractor.py
---------------------
Live feature extraction for one exam answer video.

This is the live-system replacement for windowed_extraction_v2.py.
Ported to compute the SAME features as 01_VivaAI_Feature_Extraction.ipynb
(the training notebook), so the trained GRU/Random Forest models, which
expect the transcript_only feature set, receive matching input live.

Called by app.py:
    from feature_extractor import extract_windowed
    extract_windowed(
        video_path_str  = "...",
        whisper_model   = "small.en",
        output_dir      = "...",
        window_sec      = 3.0,
        question_id     = "q1",
        expected_answer = "...",   # kept for backward-compat call signature
    )

KEY DIFFERENCE FROM TRAINING NOTEBOOK: the training notebook scored every
answer against one hardcoded EXAM_RUBRIC (a single fixed math exam shared
by all participants). The live system generates a different, report-derived
rubric per student (see generate_rubric_for_student /
extract_scoring_from_rubric in app.py, writing rubric.json + questions.json
per exam). This module reads that per-exam questions.json instead of a
hardcoded rubric; every other formula is unchanged from training.

Robustness fix carried over from windowed_extraction_v2.py: audio is
extracted to a temporary wav via ffmpeg before Whisper transcribes it, to
avoid a crash when Whisper's internal ffmpeg call chokes on malformed
webm/mp4 audio streams from the browser's MediaRecorder.
"""

import gc
import json
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import whisper

from visual_features import extract_visual_timeseries, visual_features_for_window
from audio_features import extract_audio_windows
from linguistic_features import (
    extract_linguistic_window,
    extract_answer_features,
    compute_behavioral_composites,
    build_rubric_term_regex,
    word_tokens,
)

WINDOW_SEC = 3.0
STEP_SEC = 3.0  # non-overlapping, matches training (USE_OVERLAP=False)

# Quality thresholds - must match training exactly
MIN_VIDEO_DURATION_SEC = 2.0
MIN_TRANSCRIPT_WORDS = 3
MIN_FACE_DETECTED_RATIO = 0.30
MIN_MEAN_VOICED_RATIO = 0.05

MAX_AUDIO_CLIP_RATIO = 0.01
MAX_AUDIO_LOW_VOLUME_RATIO = 0.80
MIN_AUDIO_SNR_PROXY_DB = 6.0

MIN_VIDEO_BRIGHTNESS = 25.0
MAX_VIDEO_BRIGHTNESS = 235.0
MIN_VIDEO_BLUR_SCORE = 15.0

MAX_TRANSCRIPT_LOW_CONFIDENCE_RATIO = 0.50
MAX_WHISPER_NO_SPEECH_PROB = 0.60

# Narrowed live quality-review set (see 02 notebook cell 13 / this
# session's discussion): only flag cases where evidence is genuinely
# missing, not merely lower quality (e.g. blurry webcam footage is
# normal and shouldn't trigger mandatory review).
LIVE_REVIEW_REASONS = {"low_face", "low_voice", "whisper_no_speech"}


def safe_mean(values):
    values = [
        float(v) for v in values if v is not None and np.isfinite(float(v))
    ]
    return float(np.mean(values)) if values else np.nan


def safe_max(values):
    values = [
        float(v) for v in values if v is not None and np.isfinite(float(v))
    ]
    return float(np.max(values)) if values else np.nan


def transcribe_answer(video_path, whisper_model_size="small.en"):
    """
    Transcribes one answer video with Whisper.

    Extracts audio to a temporary wav first via ffmpeg. This avoids a crash
    when Whisper's internal ffmpeg call fails on malformed mp4/webm audio
    from MediaRecorder (fix carried over from windowed_extraction_v2.py).

    Returns:
        full_text     -- complete transcript text
        words         -- list of {word, start, end, probability} dicts
        quality_meta  -- dict of whisper_avg_logprob / no_speech_prob /
                         compression_ratio, used for quality flags
    """
    wav_path = None
    transcribe_input = str(video_path)

    try:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        wav_path = tmp.name
        tmp.close()

        command = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(video_path),
            "-ac", "1", "-ar", "16000", "-f", "wav",
            wav_path,
        ]
        result = subprocess.run(command, capture_output=True)

        if result.returncode == 0 and Path(wav_path).stat().st_size > 1000:
            transcribe_input = wav_path
            print("  [Whisper] Audio extracted to wav OK")
        else:
            print("  [Whisper] Wav extraction failed, using video directly")
    except Exception as wav_err:
        print(f"  [Whisper] Wav extraction error: {wav_err}, using video directly")

    try:
        model = whisper.load_model(whisper_model_size)

        result = model.transcribe(
            transcribe_input,
            language="en",
            word_timestamps=True,
            fp16=False,
            verbose=False,
        )

        segments = result.get("segments", [])
        words = []

        for segment in segments:
            segment_words = segment.get("words")
            if segment_words:
                for word in segment_words:
                    words.append({
                        "word": str(word.get("word", "")).strip(),
                        "start": round(float(word.get("start", 0.0)), 3),
                        "end": round(float(word.get("end", 0.0)), 3),
                        "probability": (
                            round(float(word.get("probability")), 4)
                            if word.get("probability") is not None
                            else None
                        ),
                    })
            else:
                # No word-level timing available (fallback path). Split
                # the segment text into words and estimate even spacing
                # across the segment's start/end so downstream code that
                # expects a word list (word_tokens, pause_gaps, etc.)
                # still has something reasonable to work with.
                segment_text = str(segment.get("text", "")).strip()
                segment_start = float(segment.get("start", 0.0))
                segment_end = float(segment.get("end", segment_start))
                segment_words_text = segment_text.split()

                if segment_words_text:
                    span = max(segment_end - segment_start, 1e-6)
                    step = span / len(segment_words_text)
                    for i, w in enumerate(segment_words_text):
                        words.append({
                            "word": w,
                            "start": round(segment_start + i * step, 3),
                            "end": round(segment_start + (i + 1) * step, 3),
                            "probability": None,
                        })

        avg_logprob_values = [s.get("avg_logprob") for s in segments]
        no_speech_values = [s.get("no_speech_prob") for s in segments]
        compression_values = [s.get("compression_ratio") for s in segments]

        quality_meta = {
            "whisper_avg_logprob": safe_mean(avg_logprob_values),
            "whisper_no_speech_prob": safe_mean(no_speech_values),
            "whisper_no_speech_prob_max": safe_max(no_speech_values),
            "whisper_compression_ratio": safe_mean(compression_values),
        }

        full_text = str(result.get("text", "")).strip()

        del model
        gc.collect()

        return full_text, words, quality_meta

    finally:
        if wav_path and Path(wav_path).exists():
            Path(wav_path).unlink()


def get_quality_flags(
    video_duration_sec,
    transcript_word_count,
    face_ratio,
    audio_quality,
    visual_metadata,
    transcript_confidence_mean,
    transcript_low_confidence_ratio,
    whisper_no_speech_prob_max,
    video_width,
    video_height,
):
    """
    Full quality-flag set. Matches training notebook's get_quality_flags
    exactly, for consistency with training-time quality_status/quality_reason
    columns (kept for reference/analysis even though the live review
    trigger only acts on LIVE_REVIEW_REASONS).
    """
    reasons = []

    if float(video_duration_sec) < MIN_VIDEO_DURATION_SEC:
        reasons.append("short_video")

    if transcript_word_count < MIN_TRANSCRIPT_WORDS:
        reasons.append("short_transcript")

    if face_ratio < MIN_FACE_DETECTED_RATIO:
        reasons.append("low_face")

    if audio_quality["audio_mean_voiced_ratio"] < MIN_MEAN_VOICED_RATIO:
        reasons.append("low_voice")

    if audio_quality["audio_clip_ratio"] > MAX_AUDIO_CLIP_RATIO:
        reasons.append("audio_clipping")

    if audio_quality["audio_low_volume_ratio"] > MAX_AUDIO_LOW_VOLUME_RATIO:
        reasons.append("low_volume")

    if audio_quality["audio_snr_proxy_db"] < MIN_AUDIO_SNR_PROXY_DB:
        reasons.append("low_audio_range")

    brightness = visual_metadata.get("video_brightness_mean", np.nan)
    if np.isfinite(brightness):
        if brightness < MIN_VIDEO_BRIGHTNESS:
            reasons.append("dark_video")
        elif brightness > MAX_VIDEO_BRIGHTNESS:
            reasons.append("bright_video")

    blur_score = visual_metadata.get("video_blur_score", np.nan)
    if np.isfinite(blur_score) and blur_score < MIN_VIDEO_BLUR_SCORE:
        reasons.append("blurry_video")

    if (
        np.isfinite(transcript_low_confidence_ratio)
        and transcript_low_confidence_ratio > MAX_TRANSCRIPT_LOW_CONFIDENCE_RATIO
    ):
        reasons.append("low_transcript_confidence")

    if (
        np.isfinite(whisper_no_speech_prob_max)
        and whisper_no_speech_prob_max > MAX_WHISPER_NO_SPEECH_PROB
    ):
        reasons.append("whisper_no_speech")

    if int(video_width) <= 0 or int(video_height) <= 0:
        reasons.append("bad_video_size")

    quality_status = "ok" if not reasons else "check"
    quality_reason = "|".join(reasons)

    needs_live_review = bool(set(reasons) & LIVE_REVIEW_REASONS)

    return quality_status, quality_reason, needs_live_review


def extract_windowed(
    video_path_str,
    whisper_model="small.en",
    output_dir=None,
    window_sec=WINDOW_SEC,
    question_id=None,
    expected_answer=None,   # kept for backward-compat call signature; unused
    rubric_entry=None,
):
    """
    Extracts all features for one answer video: window-level rows +
    one answer-level summary row (matching training notebook shape).

    Args:
        video_path_str: path to the recorded answer video
        whisper_model:  Whisper model size (default matches training)
        output_dir:     directory to write the windowed CSV into
        window_sec:     window length in seconds (default 3.0, matches training)
        question_id:    e.g. "q1".."q5"
        rubric_entry:   dict for this question from the exam's rubric.json /
                         questions.json, REQUIRED for correctness/keyword
                         features to work. If not passed directly, the
                         caller (app.py) should read it from questions.json
                         and pass it in.

    Returns:
        windows_df   -- one row per 3-second window
        answer_row   -- one dict summarizing the whole answer

    Writes windows_df to {output_dir}/{question_id}_windowed.csv if
    output_dir is given (matches app.py's expected file naming).
    """
    video_path = Path(video_path_str)

    if rubric_entry is None:
        rubric_entry = {}

    rubric_terms = rubric_entry.get("rubric_terms")  # optional, may be absent
    rubric_term_re = build_rubric_term_regex(rubric_terms)

    print(f"  Extracting features: {video_path.name} ({question_id})")

    full_text, whisper_words, transcript_quality_meta = transcribe_answer(
        video_path, whisper_model
    )

    print("  Extracting visual timeseries...")
    visual_data, visual_metadata = extract_visual_timeseries(video_path)

    print("  Extracting audio windows...")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        temp_wav_path = Path(tmp.name)

    audio_windows, audio_quality = extract_audio_windows(video_path, temp_wav_path)

    if not audio_windows:
        raise RuntimeError("No audio windows were produced.")

    face_ratio = (
        float(np.mean(visual_data["face_detected"]))
        if len(visual_data.get("face_detected", []))
        else 0.0
    )

    transcript_word_count = len(word_tokens(full_text))

    word_probabilities = [
        float(w["probability"])
        for w in whisper_words
        if w.get("probability") is not None
    ]
    transcript_confidence_mean = (
        float(np.mean(word_probabilities)) if word_probabilities else np.nan
    )
    transcript_low_confidence_ratio = (
        float(np.mean(np.asarray(word_probabilities) < 0.50))
        if word_probabilities
        else np.nan
    )

    quality_status, quality_reason, needs_live_review = get_quality_flags(
        video_duration_sec=audio_quality["audio_duration_sec"],
        transcript_word_count=transcript_word_count,
        face_ratio=face_ratio,
        audio_quality=audio_quality,
        visual_metadata=visual_metadata,
        transcript_confidence_mean=transcript_confidence_mean,
        transcript_low_confidence_ratio=transcript_low_confidence_ratio,
        whisper_no_speech_prob_max=transcript_quality_meta["whisper_no_speech_prob_max"],
        video_width=visual_metadata["width"],
        video_height=visual_metadata["height"],
    )

    # ── Per-window features ──────────────────────────────────────────────
    window_rows = []

    for audio_win in audio_windows:
        start_time = float(audio_win["win_start"])
        end_time = float(audio_win["win_end"])

        visual_win = visual_features_for_window(visual_data, start_time, end_time)

        linguistic_win = extract_linguistic_window(
            whisper_words, start_time, end_time, rubric_entry, rubric_term_re
        )

        composite_win = compute_behavioral_composites(
            visual_win, audio_win, linguistic_win
        )

        window_row = {
            "video_id": video_path.stem,
            "question_id": question_id,
            "window_idx": int(audio_win["window_idx"]),
            "win_start": start_time,
            "win_end": end_time,
            "window_sec": window_sec,
            "step_sec": STEP_SEC,
            "quality_status": quality_status,
            "quality_reason": quality_reason,
        }
        window_row.update(visual_win)
        window_row.update({
            k: v for k, v in audio_win.items()
            if k not in {"window_idx", "win_start", "win_end"}
        })
        window_row.update(linguistic_win)
        window_row.update(composite_win)

        window_rows.append(window_row)

    # ── Answer-level features ────────────────────────────────────────────
    answer_duration = float(audio_quality["audio_duration_sec"])

    answer_features = extract_answer_features(
        full_text, whisper_words, rubric_entry, answer_duration, rubric_term_re
    )

    answer_row = {
        "video_id": video_path.stem,
        "question_id": question_id,
        "video_duration_sec": answer_duration,
        "video_width": int(visual_metadata["width"]),
        "video_height": int(visual_metadata["height"]),
        "transcript_text": full_text,
        "transcript_confidence_mean": (
            round(transcript_confidence_mean, 6)
            if np.isfinite(transcript_confidence_mean) else np.nan
        ),
        "transcript_low_confidence_ratio": (
            round(transcript_low_confidence_ratio, 6)
            if np.isfinite(transcript_low_confidence_ratio) else np.nan
        ),
        **transcript_quality_meta,
        "answer_face_detected_ratio": round(face_ratio, 6),
        "video_brightness_mean": visual_metadata["video_brightness_mean"],
        "video_brightness_std": visual_metadata["video_brightness_std"],
        "video_dark_frame_ratio": visual_metadata["video_dark_frame_ratio"],
        "video_bright_frame_ratio": visual_metadata["video_bright_frame_ratio"],
        "video_blur_score": visual_metadata["video_blur_score"],
        "video_blurry_frame_ratio": visual_metadata["video_blurry_frame_ratio"],
        "window_sec": window_sec,
        "step_sec": STEP_SEC,
        "visual_sample_fps": round(visual_metadata["sample_fps"], 4),
        "visual_sampled_frames": int(visual_metadata["sampled_frames"]),
        "quality_status": quality_status,
        "quality_reason": quality_reason,
        "needs_live_review": needs_live_review,
    }
    answer_row.update(audio_quality)
    answer_row.update(answer_features)

    windows_df = pd.DataFrame(window_rows)

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / f"{question_id}_windowed.csv"
        windows_df.to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path.name}")

        transcript_txt_path = output_dir / f"{question_id}_transcript.txt"
        with open(transcript_txt_path, "w", encoding="utf-8") as f:
            f.write(f"--- FULL TRANSCRIPT ---\n\n{full_text}\n")

    return windows_df, answer_row
