"""
linguistic_features.py
-----------------------
Linguistic, correctness, and rubric-based feature extraction from Whisper
transcripts.

Ported exactly from 01_VivaAI_Feature_Extraction.ipynb (cells 3 and 14) so
that live predictions use features computed identically to training data.

KEY DIFFERENCE FROM TRAINING NOTEBOOK: get_question_rubric() there read from
a single hardcoded EXAM_RUBRIC dict (one fixed math exam, shared by every
participant). In the live system, every student has their own report-derived
rubric (built by generate_rubric_for_student / extract_scoring_from_rubric
in app.py). Every function below that needs rubric data takes a `rubric`
dict parameter instead of reading a module-level constant. The formulas
themselves are unchanged.

Expected rubric shape (matches questions.json written by app.py):
    {
      "q1": {
        "question": "...",
        "expected_answer": "...",   # == reference_answer
        "required_values": [...],
        "keywords": [...],
      },
      ...
    }

Produces, per 3-second window:
    ling_word_count, ling_speech_rate_wpm, ling_filler_count,
    ling_filler_ratio, ling_restart_count, ling_restart_ratio,
    ling_hedge_ratio, ling_rubric_term_density, ling_causal_density,
    ling_value_mention_count, ling_structured_evidence_count,
    ling_repetition_ratio, ling_bigram_repeat_ratio, ling_pause_gap_mean,
    ling_pause_gap_max, ling_speech_rate_var, ling_rubric_keyword_hits

Produces, per whole answer:
    ans_word_count, ans_speech_rate_wpm, ans_filler_count, ans_filler_ratio,
    ans_restart_count, ans_restart_ratio, ans_hedge_count, ans_hedge_ratio,
    ans_repetition_ratio, ans_bigram_repeat_ratio, ans_sentence_repeat_ratio,
    ans_mean_word_gap, ans_max_word_gap, ans_transcript_correctness,
    ans_specificity_score, ans_rubric_term_count, ans_rubric_term_density,
    ans_causal_count, ans_causal_density, ans_reasoning_count,
    ans_reasoning_density, ans_value_mention_count,
    ans_value_mention_density, ans_structured_evidence_count,
    ans_structured_evidence_density, ans_keyword_coverage,
    ans_rubric_word_overlap, ans_required_value_match, nli_status

Also: compute_behavioral_composites() combines visual + audio + linguistic
into comp_attention_score, comp_hesitation_score, comp_delivery_variability.
"""

import re
import math

import numpy as np

# Shared regex patterns - must match training exactly

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "because", "but",
    "by", "for", "from", "had", "has", "have", "he", "her", "his",
    "i", "in", "is", "it", "its", "of", "on", "or", "she", "so",
    "that", "the", "their", "them", "there", "they", "this", "to",
    "was", "were", "which", "with", "you",
}

NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "fifty": "50",
}

FILLER_RE = re.compile(
    r"\b(um|uh|uhh|hmm|erm|you know|i mean|sort of|kind of)\b",
    re.IGNORECASE,
)

RESTART_RE = re.compile(
    r"\b(sorry|let me|actually|no wait|wait|what i meant|"
    r"correction|or rather|let me rephrase|let me start)\b",
    re.IGNORECASE,
)

HEDGE_RE = re.compile(
    r"\b(i think|maybe|not sure|i forgot|something like|"
    r"i believe|probably|i guess|it seemed|seemed|perhaps)\b",
    re.IGNORECASE,
)

CAUSAL_RE = re.compile(
    r"\b(because|since|therefore|which means|which gives|"
    r"that means|this means|thus|hence|as a result|due to|"
    r"consequently|leading to|so that|by doing|this gives)\b",
    re.IGNORECASE,
)

REASONING_RE = re.compile(
    r"\b(because|therefore|which means|this gives|after|before|"
    r"first|then|next|finally|substitut(?:e|ing|ed)|"
    r"factor(?:ing|ed)?|expand(?:ing|ed)?|simplif(?:y|ying|ied)|"
    r"solve(?:d|ing)?|divide(?:d|ing)?|multiply|subtract|add)\b",
    re.IGNORECASE,
)

VALUE_MENTION_RE = re.compile(
    r"(?<!\w)-?\d+(?:\.\d+)?(?!\w)|"
    r"\b(zero|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"twenty|thirty|forty|fifty|hundred|half|twice)\b",
    re.IGNORECASE,
)

STRUCTURED_EVIDENCE_RE = re.compile(
    r"\b(?:x|y)\s*(?:equals?|equal\s+to|is)\s*"
    r"(?:negative\s+|minus\s+)?"
    r"(?:\d+(?:\.\d+)?|zero|one|two|three|four|five|six|"
    r"seven|eight|nine|ten|fifty)\b|"
    r"\b(?:x|y)\s+squared\b|"
    r"\b(?:perfect\s+square|square\s+root|quadratic\s+formula)\b",
    re.IGNORECASE,
)

# NOTE: RUBRIC_TERM_RE is built per-rubric (see build_rubric_term_regex
# below), since the training notebook's rubric_terms list was specific to
# one fixed math exam and the live system has different report-specific
# vocabulary per student. If a rubric doesn't provide its own term list,
# this generic set is used as a fallback so the feature never breaks.
DEFAULT_RUBRIC_TERMS = [
    "identity", "infinitely", "infinite", "solution", "solutions",
    "double", "root", "roots", "quadratic", "linear", "equation",
    "inscribed", "central", "angle", "theorem", "factored",
    "repeated", "real", "negative", "squared", "simplify",
    "substitute", "expand", "expression", "variable", "coefficient",
    "exponent", "polynomial", "circle", "radius", "diameter",
    "arc", "degrees", "formula", "solve",
]

ENABLE_NLI_CORRECTNESS = True
_NLI_MODEL = None


def clean_spaces(value):
    return re.sub(r"\s+", " ", str(value)).strip()


def build_rubric_term_regex(rubric_terms=None):
    """
    Builds the whole-word regex used for ling_rubric_term_density /
    ans_rubric_term_count. Falls back to DEFAULT_RUBRIC_TERMS (the training
    exam's terms) if no report-specific term list is supplied.
    """
    terms = rubric_terms or DEFAULT_RUBRIC_TERMS
    return re.compile(
        r"\b(" + "|".join(re.escape(t) for t in terms) + r")\b",
        re.IGNORECASE,
    )


def load_nli_model():
    global _NLI_MODEL

    if _NLI_MODEL is None and ENABLE_NLI_CORRECTNESS:
        from transformers import pipeline

        print("Loading transcript correctness model...")
        _NLI_MODEL = pipeline(
            "text-classification",
            model="cross-encoder/nli-deberta-v3-small",
            device=-1,
        )

    return _NLI_MODEL


def words_in_window(words, start_time, end_time):
    return [
        word
        for word in words
        if float(word.get("start", 0.0)) < end_time
        and float(word.get("end", 0.0)) > start_time
    ]


def word_tokens(text):
    return re.findall(r"[a-zA-Z0-9']+", str(text).lower())


def content_tokens(text):
    return [
        token
        for token in word_tokens(text)
        if token not in STOP_WORDS and len(token) > 1
    ]


def pause_gaps(words):
    gaps = []
    for index in range(1, len(words)):
        gap = float(words[index].get("start", 0.0)) - float(
            words[index - 1].get("end", 0.0)
        )
        if 0.0 <= gap <= 5.0:
            gaps.append(gap)
    return gaps


def repetition_ratio(tokens):
    if not tokens:
        return 0.0
    return max(0.0, 1.0 - len(set(tokens)) / len(tokens))


def repeated_bigram_ratio(tokens):
    if len(tokens) < 2:
        return 0.0
    bigrams = list(zip(tokens[:-1], tokens[1:]))
    return max(0.0, 1.0 - len(set(bigrams)) / len(bigrams))


def repeated_sentence_ratio(text):
    sentences = [
        clean_spaces(sentence.lower())
        for sentence in re.split(r"[.!?]+", str(text))
        if clean_spaces(sentence)
    ]
    if not sentences:
        return 0.0
    return max(0.0, 1.0 - len(set(sentences)) / len(sentences))


def phrase_present(text, phrase):
    text = clean_spaces(str(text).lower())
    phrase = clean_spaces(str(phrase).lower())
    pattern = r"(?<!\w)" + re.escape(phrase).replace(r"\ ", r"\s+") + r"(?!\w)"
    return re.search(pattern, text) is not None


def keyword_hits(text, rubric_entry):
    keywords = rubric_entry.get("keywords", [])
    return sum(1 for keyword in keywords if phrase_present(text, keyword))


def keyword_coverage(text, rubric_entry):
    keywords = rubric_entry.get("keywords", [])
    if not keywords:
        return 0.0
    return keyword_hits(text, rubric_entry) / len(keywords)


def expected_word_overlap(text, rubric_entry):
    reference = rubric_entry.get("expected_answer", "") or rubric_entry.get(
        "reference_answer", ""
    )
    reference_tokens = set(content_tokens(reference))
    answer_tokens = set(content_tokens(text))

    if not reference_tokens:
        return 0.0

    return len(reference_tokens & answer_tokens) / len(reference_tokens)


def normalize_number(value):
    value = clean_spaces(str(value).lower())
    value = NUMBER_WORDS.get(value, value)
    value = value.lstrip("+")
    return value


def found_values(text):
    lowered = clean_spaces(str(text).lower())
    found = set()
    occupied_spans = []

    negative_pattern = re.compile(
        r"\b(?:negative|minus)\s+"
        r"(-?\d+(?:\.\d+)?|zero|one|two|three|four|five|"
        r"six|seven|eight|nine|ten|fifty)\b",
        re.IGNORECASE,
    )

    for match in negative_pattern.finditer(lowered):
        value = normalize_number(match.group(1))
        found.add("-" + value.lstrip("-"))
        occupied_spans.append(match.span())

    for match in VALUE_MENTION_RE.finditer(lowered):
        if any(start <= match.start() < end for start, end in occupied_spans):
            continue
        value = normalize_number(match.group(0))
        if value not in {"half", "twice"}:
            found.add(value)

    return found


def expected_value_match(text, rubric_entry):
    expected = set(str(v) for v in rubric_entry.get("required_values", []))
    if not expected:
        return 0.0
    found = found_values(text)
    return len(found & expected) / len(expected)


def transcript_correctness_score(text, rubric_entry):
    reference = rubric_entry.get("expected_answer", "") or rubric_entry.get(
        "reference_answer", ""
    )

    if (
        not ENABLE_NLI_CORRECTNESS
        or len(word_tokens(text)) < 3
        or not reference
    ):
        return 0.5, "not_evaluated"

    try:
        model = load_nli_model()

        result = model(
            {"text": reference[:700], "text_pair": str(text)[:700]},
            top_k=None,
        )

        if result and isinstance(result[0], list):
            result = result[0]

        scores = {
            str(item["label"]).lower(): float(item["score"]) for item in result
        }

        entailment = max(
            [value for label, value in scores.items() if "entail" in label]
            or [0.33]
        )
        contradiction = max(
            [value for label, value in scores.items() if "contrad" in label]
            or [0.33]
        )

        score = np.clip(entailment - 0.5 * contradiction, 0.0, 1.0)
        return round(float(score), 6), "success"

    except Exception as exc:
        print("Correctness model warning:", repr(exc))
        return 0.5, "failed"


def speech_rate_variability(selected_words, start_time, end_time, bin_seconds=1.0):
    duration = max(float(end_time) - float(start_time), 1e-6)
    bin_count = max(1, int(math.ceil(duration / bin_seconds)))

    counts = np.zeros(bin_count, dtype=float)

    for word in selected_words:
        midpoint = (
            float(word.get("start", 0.0)) + float(word.get("end", 0.0))
        ) / 2.0
        relative_time = midpoint - float(start_time)
        index = int(
            np.clip(math.floor(relative_time / bin_seconds), 0, bin_count - 1)
        )
        counts[index] += 1.0

    rates_wpm = counts / bin_seconds * 60.0
    return float(np.std(rates_wpm)) if len(rates_wpm) >= 2 else 0.0


def extract_linguistic_window(
    whisper_words, start_time, end_time, rubric_entry, rubric_term_re
):
    """
    Computes ling_* features for one 3-second window.

    Args:
        whisper_words:  list of word dicts with start/end/word from Whisper
        start_time:     window start (seconds)
        end_time:       window end (seconds)
        rubric_entry:   dict for this question (question/expected_answer/
                        required_values/keywords), from live rubric.json
        rubric_term_re: compiled regex from build_rubric_term_regex()
    """
    selected_words = words_in_window(whisper_words, start_time, end_time)

    text = " ".join(
        str(word.get("word", "")).strip() for word in selected_words
    ).strip()

    tokens = word_tokens(text)
    word_count = len(tokens)
    duration = max(end_time - start_time, 1e-6)

    filler_count = len(FILLER_RE.findall(text))
    restart_count = len(RESTART_RE.findall(text))
    hedge_count = len(HEDGE_RE.findall(text))
    rubric_term_count = len(rubric_term_re.findall(text))
    causal_count = len(CAUSAL_RE.findall(text))
    value_mention_count = len(VALUE_MENTION_RE.findall(text))
    structured_evidence_count = len(STRUCTURED_EVIDENCE_RE.findall(text))
    gaps = pause_gaps(selected_words)

    return {
        "ling_word_count": int(word_count),
        "ling_speech_rate_wpm": round(word_count / duration * 60.0, 5),
        "ling_filler_count": int(filler_count),
        "ling_filler_ratio": round(filler_count / max(word_count, 1), 6),
        "ling_restart_count": int(restart_count),
        "ling_restart_ratio": round(restart_count / max(word_count, 1), 6),
        "ling_hedge_ratio": round(hedge_count / max(word_count, 1), 6),
        "ling_rubric_term_density": round(
            rubric_term_count / max(word_count, 1), 6
        ),
        "ling_causal_density": round(causal_count / max(word_count, 1), 6),
        "ling_value_mention_count": int(value_mention_count),
        "ling_structured_evidence_count": int(structured_evidence_count),
        "ling_repetition_ratio": round(repetition_ratio(tokens), 6),
        "ling_bigram_repeat_ratio": round(repeated_bigram_ratio(tokens), 6),
        "ling_pause_gap_mean": round(float(np.mean(gaps)) if gaps else 0.0, 6),
        "ling_pause_gap_max": round(float(np.max(gaps)) if gaps else 0.0, 6),
        "ling_speech_rate_var": round(
            speech_rate_variability(selected_words, start_time, end_time), 6
        ),
        "ling_rubric_keyword_hits": int(keyword_hits(text, rubric_entry)),
    }


def extract_answer_features(
    full_text, whisper_words, rubric_entry, answer_duration, rubric_term_re
):
    """
    Computes ans_* features for the whole answer (all windows combined).

    Args:
        full_text:       complete transcript text for this answer
        whisper_words:   list of word dicts with start/end/word
        rubric_entry:    dict for this question, from live rubric.json
        answer_duration: total answer duration (seconds)
        rubric_term_re:  compiled regex from build_rubric_term_regex()
    """
    tokens = word_tokens(full_text)
    word_count = len(tokens)

    filler_count = len(FILLER_RE.findall(full_text))
    restart_count = len(RESTART_RE.findall(full_text))
    hedge_count = len(HEDGE_RE.findall(full_text))
    rubric_term_count = len(rubric_term_re.findall(full_text))
    causal_count = len(CAUSAL_RE.findall(full_text))
    reasoning_count = len(REASONING_RE.findall(full_text))
    value_mention_count = len(VALUE_MENTION_RE.findall(full_text))
    structured_evidence_count = len(STRUCTURED_EVIDENCE_RE.findall(full_text))
    gaps = pause_gaps(whisper_words)

    specificity = (rubric_term_count + 1.0) / (rubric_term_count + hedge_count + 2.0)

    correctness, nli_status = transcript_correctness_score(full_text, rubric_entry)

    return {
        "ans_word_count": int(word_count),
        "ans_speech_rate_wpm": round(
            word_count / max(answer_duration, 1e-6) * 60.0, 5
        ),
        "ans_filler_count": int(filler_count),
        "ans_filler_ratio": round(filler_count / max(word_count, 1), 6),
        "ans_restart_count": int(restart_count),
        "ans_restart_ratio": round(restart_count / max(word_count, 1), 6),
        "ans_hedge_count": int(hedge_count),
        "ans_hedge_ratio": round(hedge_count / max(word_count, 1), 6),
        "ans_repetition_ratio": round(repetition_ratio(tokens), 6),
        "ans_bigram_repeat_ratio": round(repeated_bigram_ratio(tokens), 6),
        "ans_sentence_repeat_ratio": round(
            repeated_sentence_ratio(full_text), 6
        ),
        "ans_mean_word_gap": round(float(np.mean(gaps)) if gaps else 0.0, 6),
        "ans_max_word_gap": round(float(np.max(gaps)) if gaps else 0.0, 6),
        "ans_transcript_correctness": correctness,
        "ans_specificity_score": round(specificity, 6),
        "ans_rubric_term_count": int(rubric_term_count),
        "ans_rubric_term_density": round(
            rubric_term_count / max(word_count, 1), 6
        ),
        "ans_causal_count": int(causal_count),
        "ans_causal_density": round(causal_count / max(word_count, 1), 6),
        "ans_reasoning_count": int(reasoning_count),
        "ans_reasoning_density": round(reasoning_count / max(word_count, 1), 6),
        "ans_value_mention_count": int(value_mention_count),
        "ans_value_mention_density": round(
            value_mention_count / max(word_count, 1), 6
        ),
        "ans_structured_evidence_count": int(structured_evidence_count),
        "ans_structured_evidence_density": round(
            structured_evidence_count / max(word_count, 1), 6
        ),
        "ans_keyword_coverage": round(
            keyword_coverage(full_text, rubric_entry), 6
        ),
        "ans_rubric_word_overlap": round(
            expected_word_overlap(full_text, rubric_entry), 6
        ),
        "ans_required_value_match": round(
            expected_value_match(full_text, rubric_entry), 6
        ),
        "nli_status": nli_status,
    }


def compute_behavioral_composites(visual_features, audio_features, linguistic_features):
    attention = (
        0.35 * visual_features["vis_face_detected_ratio"]
        + 0.25 * visual_features["vis_gaze_stability"]
        + 0.20 * (1.0 - visual_features["vis_gaze_offcenter_ratio"])
        + 0.20 * (1.0 - visual_features["vis_head_down_ratio"])
    )

    hesitation = (
        0.40 * audio_features["aud_pause_ratio"]
        + 0.25 * min(linguistic_features["ling_filler_ratio"] / 0.20, 1.0)
        + 0.20 * min(linguistic_features["ling_restart_ratio"] / 0.10, 1.0)
        + 0.15 * min(audio_features["aud_voice_jitter"] / 0.10, 1.0)
    )

    delivery_variability = (
        0.30 * min(audio_features["aud_pitch_std"] / 80.0, 1.0)
        + 0.25 * min(audio_features["aud_energy_std"] / 0.05, 1.0)
        + 0.25 * min(visual_features["vis_head_motion_mean"] / 10.0, 1.0)
        + 0.20 * min(visual_features["vis_facial_activity"] / 0.05, 1.0)
    )

    return {
        "comp_attention_score": round(float(np.clip(attention, 0.0, 1.0)), 6),
        "comp_hesitation_score": round(float(np.clip(hesitation, 0.0, 1.0)), 6),
        "comp_delivery_variability": round(
            float(np.clip(delivery_variability, 0.0, 1.0)), 6
        ),
    }
