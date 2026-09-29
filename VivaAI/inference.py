"""
inference.py
------------
Loads the trained GRU (primary) and Random Forest (disagreement-check)
models and produces a free-rider / authentic prediction for one student's
completed exam.

Replaces the old Transformer-based inference.py. Verified directly against
the actual saved model files:
    transcript_baseline.joblib               -- Random Forest pipeline
    transcript_temporal.pt                   -- GRU checkpoint
    transcript_temporal_preprocessing.joblib -- imputer + scaler for GRU

Both models were trained on the same 78 transcript_only features, in the
same order (confirmed identical across all three files).

Architecture (matches 02_VivaAI_Training_and_Evaluation.ipynb):
    - GRU is the PRIMARY decision -- its probability is the one used.
    - Random Forest is used ONLY as an independent second opinion for the
      model_disagreement review trigger. It is never averaged into GRU's
      probability (an averaged ensemble was tested and found less stable
      than GRU alone across seeds).
    - mandatory_review fires when ANY of:
        1. model_disagreement   -- RF and GRU predict different classes
        2. gray_zone            -- GRU probability is 0.40-0.60
        3. quality_flag         -- recording has low_face / low_voice /
                                    whisper_no_speech (narrowed set, see
                                    feature_extractor.py's
                                    LIVE_REVIEW_REASONS)

Called by app.py after all 5 answer CSVs exist for an exam:
    from inference import VivaAIPredictor
    predictor = VivaAIPredictor()
    result = predictor.predict_student(windowed_csv_paths, quality_flags)
"""

import os
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

MODEL_DIR = Path(__file__).parent / "models"

RF_PATH          = MODEL_DIR / "transcript_baseline.joblib"
GRU_PATH          = MODEL_DIR / "transcript_temporal.pt"
GRU_PREPROC_PATH  = MODEL_DIR / "transcript_temporal_preprocessing.joblib"

GRAY_ZONE_LOW  = 0.40
GRAY_ZONE_HIGH = 0.60

QUESTION_ORDER = ["Q1", "Q2", "Q3", "Q4", "Q5"]


# ── GRU MODEL DEFINITION (must match training notebook exactly) ────────────

class GRUClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=32, dropout=0.20):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, hidden = self.gru(x)
        hidden = self.dropout(hidden[-1])
        return self.output(hidden).squeeze(1)


# ── PREDICTOR CLASS ──────────────────────────────────────────────────────────

class VivaAIPredictor:
    """
    Loads both trained models once and runs the full two-model, three-trigger
    review policy on a completed student exam (5 answer CSVs).
    """

    def __init__(
        self,
        rf_path=RF_PATH,
        gru_path=GRU_PATH,
        gru_preproc_path=GRU_PREPROC_PATH,
    ):
        self.features = None
        self.threshold = 0.5

        self.rf_model = None

        self.gru_model = None
        self.gru_imputer = None
        self.gru_scaler = None

        self._load(rf_path, gru_path, gru_preproc_path)

    def _load(self, rf_path, gru_path, gru_preproc_path):
        for path, label in [
            (rf_path, "Random Forest model"),
            (gru_path, "GRU checkpoint"),
            (gru_preproc_path, "GRU preprocessing bundle"),
        ]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"{label} not found: {path}")

        # ── Random Forest ──────────────────────────────────────────────
        self.rf_model = joblib.load(rf_path)
        rf_features = list(self.rf_model.feature_names_in_)

        # ── GRU checkpoint ──────────────────────────────────────────────
        checkpoint = torch.load(gru_path, map_location="cpu", weights_only=False)

        self.features = checkpoint["features"]
        self.threshold = checkpoint.get("threshold", 0.5)

        if rf_features != self.features:
            raise RuntimeError(
                "Random Forest and GRU were trained on different feature "
                "sets/order, this should never happen for transcript_only "
                "models. Check that both files came from the same notebook run."
            )

        self.gru_model = GRUClassifier(
            input_size=checkpoint["input_size"],
            hidden_size=checkpoint.get("hidden_size", 32),
            dropout=checkpoint.get("dropout", 0.2),
        )
        self.gru_model.load_state_dict(checkpoint["state_dict"])
        self.gru_model.eval()

        # ── GRU preprocessing (imputer + scaler, fit at training time) ──
        preproc = joblib.load(gru_preproc_path)
        self.gru_imputer = preproc["imputer"]
        self.gru_scaler = preproc["scaler"]

        if preproc["features"] != self.features:
            raise RuntimeError(
                "GRU preprocessing bundle's feature list does not match "
                "the GRU checkpoint's feature list."
            )

        print(f"Loaded RF + GRU models: {len(self.features)} features")
        print(f"GRU threshold: {self.threshold}")

    def _load_answer_features(self, windowed_csv_path):
        """
        Loads one answer's windowed CSV and returns a single answer-level
        feature row (transcript_only expects answer-level ans_*/ling_*
        aggregate columns, one row per answer, NOT per-window rows).

        feature_extractor.py's extract_windowed() writes both window-level
        rows (used only for ling_*__mean/__std/__slope aggregation) and the
        answer-level summary. Training aggregated ling_*/aud_*/vis_*/comp_*
        window columns into __mean/__std/__slope per answer; this must be
        replicated here exactly the same way for feature_extractor's raw
        windowed CSV output.
        """
        windows = pd.read_csv(windowed_csv_path)

        if windows.empty:
            raise ValueError(f"Windowed CSV is empty: {windowed_csv_path}")

        # Any column present on every window row that isn't identification/
        # metadata is aggregated the same way training aggregated window
        # columns: __mean, __std, __slope (mean-centered least-squares).
        window_prefixes = ("aud_", "vis_", "ling_", "comp_")
        window_feature_cols = [
            c for c in windows.columns
            if c.startswith(window_prefixes)
            and pd.api.types.is_numeric_dtype(windows[c])
        ]

        row = {}

        if "win_start" in windows.columns:
            time_values = pd.to_numeric(
                windows["win_start"], errors="coerce"
            ).to_numpy(dtype=float)
        else:
            time_values = np.arange(len(windows), dtype=float)

        for feature in window_feature_cols:
            values = pd.to_numeric(
                windows[feature], errors="coerce"
            ).to_numpy(dtype=float)
            finite = values[np.isfinite(values)]

            if finite.size:
                row[f"{feature}__mean"] = float(np.mean(finite))
                row[f"{feature}__std"] = float(np.std(finite, ddof=0))
                row[f"{feature}__slope"] = self._safe_slope(time_values, values)
            else:
                row[f"{feature}__mean"] = np.nan
                row[f"{feature}__std"] = np.nan
                row[f"{feature}__slope"] = np.nan

        return row

    @staticmethod
    def _safe_slope(x_values, y_values):
        """Mean-centered least-squares slope. Matches training notebook exactly."""
        x = np.asarray(x_values, dtype=float)
        y = np.asarray(y_values, dtype=float)

        valid = np.isfinite(x) & np.isfinite(y)
        if valid.sum() < 2:
            return 0.0

        x = x[valid]
        y = y[valid]

        if np.unique(x).size < 2:
            return 0.0

        x = x - x.mean()
        denominator = float(np.sum(x ** 2))

        if denominator <= 0:
            return 0.0

        return float(np.sum(x * (y - y.mean())) / denominator)

    def _build_feature_row(self, windowed_csv_path, answer_row=None):
        """
        Builds one row of the 78 transcript_only features for one answer.

        answer_row (optional): the answer-level dict already returned by
        feature_extractor.extract_windowed(), which contains the real ans_*
        features directly (word count, correctness, keyword coverage, etc.)
        computed once over the whole answer, not re-derived from windows.
        If not supplied, only the window-aggregated ling_*__mean/__std/__slope
        features will be available and ans_* columns will be NaN (imputed).
        """
        row = self._load_answer_features(windowed_csv_path)

        if answer_row:
            for key, value in answer_row.items():
                if key.startswith("ans_"):
                    row[key] = value

        return row

    def predict_answer_row(self, feature_row):
        """
        Runs RF + GRU on a single answer's feature row (not yet used for a
        final decision, since GRU needs all 5 answers as a sequence). This is
        exposed for debugging/inspection of individual answers.
        """
        df = pd.DataFrame([feature_row])[self.features]
        rf_prob = float(self.rf_model.predict_proba(df)[:, 1][0])
        return rf_prob

    def predict_student(self, windowed_csv_paths, answer_rows=None, quality_flags=None):
        """
        Produces the final prediction for one student's completed exam.

        Args:
            windowed_csv_paths: list of 5 paths to {q1..q5}_windowed.csv,
                                 in Q1..Q5 order
            answer_rows:         optional list of 5 answer-level dicts
                                  (from feature_extractor.extract_windowed's
                                  second return value), supplies the real
                                  ans_* features directly
            quality_flags:       optional list of 5 booleans
                                  (needs_live_review from feature_extractor),
                                  one per question

        Returns:
            dict with:
                gru_probability      -- GRU's P(authentic), the primary score
                gru_prediction       -- "authentic" or "free_rider"
                rf_probability       -- Random Forest's P(authentic)
                rf_prediction        -- "authentic" or "free_rider"
                models_agree         -- bool
                model_disagreement   -- bool
                gray_zone            -- bool (GRU probability in 0.40-0.60)
                quality_flag         -- bool (any answer flagged for review)
                mandatory_review     -- bool (any of the three triggers fired)
                confidence           -- abs(gru_probability - 0.5) * 2
                features_used        -- list of feature names
        """
        if len(windowed_csv_paths) != 5:
            raise ValueError(
                f"Expected exactly 5 answer CSVs (Q1-Q5), got "
                f"{len(windowed_csv_paths)}. GRU requires a complete "
                f"5-question sequence and cannot run on a partial exam."
            )

        answer_rows = answer_rows or [None] * 5
        quality_flags = quality_flags or [False] * 5

        feature_rows = [
            self._build_feature_row(path, answer_row)
            for path, answer_row in zip(windowed_csv_paths, answer_rows)
        ]

        # ── Random Forest: one prediction per answer, averaged ──────────
        # (matches training's session-level aggregation: mean probability
        # across the 5 answers, thresholded at 0.5)
        rf_df = pd.DataFrame(feature_rows)[self.features]
        rf_probs = self.rf_model.predict_proba(rf_df)[:, 1]
        rf_probability = float(np.mean(rf_probs))
        rf_prediction = "authentic" if rf_probability >= 0.5 else "free_rider"

        # ── GRU: full 5-step sequence, imputed + scaled with training's
        # fitted parameters ──────────────────────────────────────────────
        gru_df = pd.DataFrame(feature_rows)[self.features]
        imputed = self.gru_imputer.transform(gru_df)
        scaled = self.gru_scaler.transform(imputed)

        sequence = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0)  # (1, 5, 78)

        with torch.no_grad():
            logit = self.gru_model(sequence).squeeze()
            gru_probability = float(torch.sigmoid(logit).item())

        gru_prediction = (
            "authentic" if gru_probability >= self.threshold else "free_rider"
        )

        # ── Review policy ─────────────────────────────────────────────
        models_agree = rf_prediction == gru_prediction
        model_disagreement = not models_agree

        gray_zone = GRAY_ZONE_LOW <= gru_probability <= GRAY_ZONE_HIGH

        quality_flag = any(bool(q) for q in quality_flags)

        mandatory_review = model_disagreement or gray_zone or quality_flag

        confidence = abs(gru_probability - 0.5) * 2.0

        return {
            "gru_probability": round(gru_probability, 4),
            "gru_prediction": gru_prediction,
            "rf_probability": round(rf_probability, 4),
            "rf_prediction": rf_prediction,
            "models_agree": models_agree,
            "model_disagreement": model_disagreement,
            "gray_zone": gray_zone,
            "quality_flag": quality_flag,
            "mandatory_review": mandatory_review,
            "confidence": round(confidence, 4),
            # Primary output fields, matches old inference.py's shape so
            # app.py's existing result-saving code needs minimal changes.
            "authentic_probability": round(gru_probability, 4),
            "prediction": gru_prediction,
            "features_used": self.features,
        }


# ── QUICK TEST ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    predictor = VivaAIPredictor()
    print("Predictor loaded successfully.")
    print(f"Features ({len(predictor.features)}): {predictor.features[:5]}...")
    print(f"GRU threshold: {predictor.threshold}")
