# VivaAI

VivaAI is an AI-powered oral examination system designed to detect free riders
in university group projects. Students record short video responses to
AI-generated questions about their submitted work. The system extracts
behavioral, acoustic, and linguistic features from each response and uses two
independent models, a GRU (primary decision) and a Random Forest (agreement
check), to classify each student as authentic or a free rider.

Cases where the two models disagree, where the GRU's confidence falls in a
gray zone, or where recording quality is poor are automatically flagged for
manual review rather than resolved automatically. The instructor reviews
flagged cases, watches the recordings, reads the transcripts, and makes the
final decision.

---

## Table of Contents

1. [How It Works](#how-it-works)
2. [Project Structure](#project-structure)
3. [Requirements](#requirements)
4. [Setup](#setup)
5. [Running the Application](#running-the-application)
6. [Features Extracted](#features-extracted)
7. [Model Architecture](#model-architecture)

---

## How It Works

1. An instructor creates a course, adds groups and students, and creates a submission slot.
2. Each student uploads their project report (PDF/DOCX) through the student portal.
3. The system generates 5 personalized oral exam questions using GPT, based on the content of the uploaded report, along with a scoring rubric.
4. Each student records a short video response to each question through the live exam interface. The AI examiner reads each question aloud using a synthesized voice.
5. Once all 5 responses are recorded, the system extracts visual (MediaPipe), acoustic (Librosa), and linguistic (Whisper transcript) features from each response.
6. A trained GRU model classifies the student as authentic or free rider based on the full 5-question sequence. A Random Forest model runs independently as a second opinion.
7. If the two models disagree, the GRU's confidence is in the 0.40-0.60 gray zone, or the recording quality is poor, the result is automatically flagged for manual review.
8. The instructor reviews the AI's decision (and the reason for any flag), watches the recordings, reads the transcripts, and makes the final grading decision.
9. Grades are saved and released to students through the student portal.

---

## Project Structure

```
VivaAI/
|
├── app.py                        Flask backend: all API routes and page rendering
├── inference.py                  Loads the trained GRU + Random Forest models and runs predictions
├── feature_extractor.py          Live feature extraction: orchestrates Whisper transcription,
│                                  quality flagging, and per-question windowed CSV output
├── audio_features.py             Audio feature extraction (pitch, energy, pauses) via Librosa
├── visual_features.py            Visual feature extraction (blink, gaze, head pose) via MediaPipe
├── linguistic_features.py        Linguistic feature extraction from Whisper transcripts
│
├── templates/
│   ├── login.html                Login page (student / instructor)
│   ├── student_portal.html       Student submission and exam portal
│   ├── viva_exam.html            Live exam recording interface
│   ├── instructor_dashboard.html Instructor course, group, and exam management
│   ├── review_page.html          Instructor review of individual student results
│   └── admin.html                Admin panel: manage all accounts and courses
│
├── static/
│   ├── css/                      Stylesheets for all pages
│   └── js/
│       └── firebase.js           Firebase config and shared auth functions
│
├── data/
│   └── exam_videos/              Live exam recordings organized by exam ID
│       └── {examId}/
│           ├── q1.webm / q1.mp4  Raw and converted video per question
│           ├── q1_windowed.csv   Extracted features per question
│           ├── q1_transcript.txt Whisper transcript per question
│           └── questions.json    GPT-generated questions, rubric, and expected answers
│
├── models/
│   ├── transcript_baseline.joblib               Trained Random Forest pipeline
│   ├── transcript_temporal.pt                   Trained GRU checkpoint
│   └── transcript_temporal_preprocessing.joblib  Imputer + scaler fit at training time
│
├── face_landmarker.task          MediaPipe face model (download separately)
├── requirements.txt
└── README.md
```

Note: the offline training/evaluation pipeline (feature extraction from raw
labeled videos, model training, cross-validation) is documented separately in
the project's training notebooks and is not part of this live application
folder.

---

## Requirements

### Python

Python 3.10 or later is required.

### System Dependencies

These must be installed separately before running `pip install -r requirements.txt`.

**ffmpeg** - required for audio extraction and video re-encoding.

```bash
# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html and add to PATH
```

**MediaPipe face model** - required by `visual_features.py`.

```bash
curl -O https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

Place `face_landmarker.task` in the project root directory.

### Python Packages

```bash
pip install -r requirements.txt
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-org/vivaai.git
cd vivaai
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the MediaPipe face model

```bash
curl -O https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

### 4. Create the environment file

Create a file named `.env` in the project root:

```
OPENAI_API_KEY=your_openai_api_key_here
FIREBASE_CREDENTIALS=serviceAccountKey.json
SECRET_KEY=any_random_secret_string
```

- `OPENAI_API_KEY` - required for GPT-based question generation and TTS.
- `FIREBASE_CREDENTIALS` - path to your Firebase service account JSON, downloaded from Firebase Console > Project Settings > Service Accounts.
- `SECRET_KEY` - used by Flask to sign session cookies. Generate one with:
  ```bash
  python3 -c "import secrets; print(secrets.token_hex(32))"
  ```

### 5. Set up Firebase

- Create a Firebase project and enable **Firestore** and **Firebase Storage**.
- Download the service account key and save it as `serviceAccountKey.json` in the project root (this file is gitignored and must never be committed).
- Set Firestore security rules to allow authenticated access:

```
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {
    match /{document=**} {
      allow read, write: if request.auth != null;
    }
  }
}
```

### 6. Add the trained models

Place the three trained model files in `models/`:
- `transcript_baseline.joblib`
- `transcript_temporal.pt`
- `transcript_temporal_preprocessing.joblib`

### 7. Create the admin account

1. Sign up at `/login` with your admin email.
2. Go to Firebase Console > Firestore > `users` collection.
3. Find the new account's document and change `role` to `"admin"`.
4. From now on, manage all instructor and student accounts from `/admin`.

---

## Running the Application

```bash
python app.py
```

The application will be available at `http://localhost:5000`.

| URL | Who uses it |
|---|---|
| `/login` | Students and instructors |
| `/student` | Student portal: submit files, take exam, view results |
| `/instructor` | Instructor dashboard: manage courses, groups, submissions |
| `/review` | Instructor review of individual student recordings and scores |
| `/admin` | Admin panel: manage all accounts (admin credentials required) |

---

## Features Extracted

Each answer is processed in non-overlapping 3-second windows. Features are
extracted per window, then aggregated to answer-level `__mean`, `__std`, and
`__slope` statistics, producing 78 `transcript_only` features total that are
fed into the GRU and Random Forest models.

### Visual Features

Extracted using MediaPipe face landmarker (`visual_features.py`):
`vis_face_detected_ratio`, `vis_off_camera_ratio`, `vis_blink_frequency`,
`vis_head_motion_mean`, `vis_head_pose_variance`, `vis_gaze_stability`,
`vis_gaze_offcenter_ratio`, `vis_head_down_ratio`, `vis_facial_activity`.

### Audio Features

Extracted using Librosa (`audio_features.py`), per 3-second window:
`aud_pitch_mean`, `aud_pitch_std`, `aud_pitch_range`, `aud_pitch_slope`,
`aud_energy_mean`, `aud_energy_std`, `aud_energy_slope`, `aud_voice_jitter`,
`aud_shimmer`, `aud_pause_ratio`, `aud_pause_count`,
`aud_mean_pause_duration`, `aud_total_pause_duration`, `aud_voiced_ratio`,
`aud_zcr_mean`.

Whole-answer audio quality stats (`audio_duration_sec`, `audio_silence_ratio`,
`audio_clip_ratio`, etc.) are also computed and used for the live quality-flag
review trigger, separately from the model input features.

### Linguistic Features

Computed from Whisper transcripts (`linguistic_features.py`), covering word
count, filler/hedge ratios, keyword and required-value coverage against the
question's rubric, pause-gap timing, and speech-rate variability.

---

## Model Architecture

VivaAI uses two independent models rather than a single classifier:

- **GRU (primary decision)** - a single-layer GRU (hidden size 32) that
  consumes the full 5-question sequence (78 features per question) and
  produces the primary authentic / free-rider probability. This is the
  score shown to instructors and used for the main decision.
- **Random Forest (agreement check)** - trained on the same 78 features,
  used only to independently verify the GRU's call. Its prediction is never
  averaged into the GRU's probability; a prior averaged-ensemble approach
  was tested and found less stable across seeds than the GRU alone.

### Mandatory Review Policy

A result is automatically flagged for manual review if any of the following
are true:

1. **Model disagreement** - the GRU and Random Forest predict different classes.
2. **Gray zone** - the GRU's probability falls between 0.40 and 0.60.
3. **Quality flag** - the recording has low face-detection ratio, low voiced-audio
   ratio, or a high Whisper no-speech confidence on any question.

Flagged results are never treated as a confident free-rider call. The
`Flagged Students` view on the instructor dashboard only shows students where
the GRU predicted free-rider *and* the Random Forest agreed *and* the case is
not in the gray zone, that is, a resolved, confident call, not merely a
case that needed a second look.

### Known Limitations

- The model was trained on real recorded answers averaging roughly 12.5
  seconds; live exam answers are capped at a matching duration during
  feature extraction so window counts stay consistent with the training
  distribution.
- Question 5 shows systematically lower accuracy than other questions, tied
  to stricter rubric vocabulary requirements. This is a known, documented
  limitation rather than a pipeline bug.
- The review policy catches model disagreement, low confidence, and poor
  recording quality, but cannot catch cases where the model is confidently
  wrong. Roughly a third of the model's overall errors are caught by the
  review policy; the rest pass through automatically and rely on the
  instructor's own judgment if they choose to spot-check clean results.
