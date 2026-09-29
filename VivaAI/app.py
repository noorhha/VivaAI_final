"""
VivaAI - Flask Backend v2
==========================
AI-powered oral examination system for detecting free riders in group projects.

Changes from v1:
  - Question generation now produces expected_answer + keywords per question
  - Added /api/video/upload route, receives video, runs feature_extractor
  - run_pipeline_for_exam now processes real windowed CSVs from uploaded videos
  - Answer time changed to 15 seconds, matching TRAINING_MATCH_DURATION_SEC
    in audio_features.py
  - Report URL is required for question generation

SETUP:
  1. pip install -r requirements.txt
  2. Create a .env file with:
       OPENAI_API_KEY=your_key_here
       FIREBASE_CREDENTIALS=path/to/serviceAccountKey.json
  3. python app.py
  4. Open http://localhost:5000
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "pipeline"))

from flask import Flask, render_template, request, jsonify, redirect
from flask_cors import CORS
from dotenv import load_dotenv
import os, threading, json, traceback, re
from pathlib import Path
from openai import OpenAI

print("=== APP.PY VERSION CHECK: FILE-STABILITY FIX ACTIVE ===")

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'vivaai-dev-secret')
CORS(app)

# Directory to store uploaded exam videos and windowed CSVs
UPLOAD_DIR = os.path.join("data", "exam_videos")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ── Firebase Admin SDK ────────────────────────────────────────────────────────
import firebase_admin
from firebase_admin import credentials, firestore

def get_firestore():
    cred_path = os.environ.get('FIREBASE_CREDENTIALS', 'serviceAccountKey.json')
    if not firebase_admin._apps:
        if os.path.exists(cred_path):
            cred = credentials.Certificate(cred_path)
            try:
                firebase_admin.initialize_app(cred)
            except ValueError:
                pass
        else:
            print(f"WARNING: Firebase credentials not found at '{cred_path}'.")
            return None
    return firestore.client()


# ── PAGE ROUTES ───────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('login.html')

@app.route('/login')
def login_page():
    return render_template('login.html')

@app.route('/student')
def student_portal():
    return render_template('student_portal.html')

@app.route('/instructor')
def instructor_dashboard():
    return render_template('instructor_dashboard.html')

@app.route('/exam')
def viva_exam():
    return render_template('viva_exam.html')

@app.route('/exam/complete')
def exam_complete():
    return redirect('/student')

@app.route('/review')
def review_page():
    return render_template('review_page.html')

# ── TTS ENDPOINT ──────────────────────────────────────────────────────────────
@app.route('/api/tts', methods=['POST'])
def text_to_speech():
    """Convert text to speech using OpenAI TTS API."""
    try:
        data  = request.get_json()
        text  = data.get('text', '')
        voice = data.get('voice', 'onyx')  # nova, alloy, echo, fable, onyx, shimmer (onyx/echo are male-sounding)

        if not text:
            return jsonify({'error': 'No text provided'}), 400

        client   = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
        response = client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=text,
        )

        from flask import Response
        return Response(
            response.content,
            mimetype='audio/mpeg',
            headers={'Content-Disposition': 'inline; filename=speech.mp3'}
        )
    except Exception as e:
        print(f"TTS error: {e}")
        return jsonify({'error': str(e)}), 500



@app.route('/admin')
def admin_page():
    return render_template('admin.html')


# ── ADMIN API ─────────────────────────────────────────────────────────────────

@app.route('/api/admin/users', methods=['GET'])
def api_admin_get_users():
    """Return all users (instructors + students)."""
    db = get_firestore()
    if not db:
        return jsonify({'error': 'Firestore unavailable'}), 500
    try:
        snap = db.collection('users').stream()
        users = []
        for doc in snap:
            d = doc.to_dict()
            d['uid'] = doc.id
            users.append(d)
        return jsonify({'users': users})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/admin/create_instructor', methods=['POST'])
def api_admin_create_instructor():
    """Create a new instructor account via Firebase Auth + Firestore."""
    data     = request.json or {}
    email    = data.get('email', '').strip()
    password = data.get('password', '').strip()
    name     = data.get('name', '').strip()

    if not email or not password or not name:
        return jsonify({'error': 'email, password, and name are required'}), 400

    try:
        from firebase_admin import auth as fb_auth
        user = fb_auth.create_user(email=email, password=password, display_name=name)
        db   = get_firestore()
        if db:
            db.collection('users').document(user.uid).set({
                'name':      name,
                'email':     email,
                'role':      'instructor',
                'createdAt': firestore.SERVER_TIMESTAMP,
            })
        return jsonify({'success': True, 'uid': user.uid})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/admin/create_student', methods=['POST'])
def api_admin_create_student():
    """Create a new student account."""
    data      = request.json or {}
    email     = data.get('email', '').strip()
    password  = data.get('password', '').strip()
    name      = data.get('name', '').strip()
    studentId = data.get('studentId', '').strip()

    if not email or not password or not name:
        return jsonify({'error': 'email, password, and name are required'}), 400

    try:
        from firebase_admin import auth as fb_auth
        user = fb_auth.create_user(email=email, password=password, display_name=name)
        db   = get_firestore()
        if db:
            db.collection('users').document(user.uid).set({
                'name':      name,
                'email':     email,
                'studentId': studentId,
                'role':      'student',
                'createdAt': firestore.SERVER_TIMESTAMP,
            })
        return jsonify({'success': True, 'uid': user.uid})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/admin/delete_user', methods=['POST'])
def api_admin_delete_user():
    """Delete a user from Firebase Auth and Firestore."""
    data = request.json or {}
    uid  = data.get('uid', '').strip()
    if not uid:
        return jsonify({'error': 'uid is required'}), 400
    try:
        from firebase_admin import auth as fb_auth
        fb_auth.delete_user(uid)
        db = get_firestore()
        if db:
            db.collection('users').document(uid).delete()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/admin/courses', methods=['GET'])
def api_admin_get_courses():
    """Return all courses with instructor info."""
    db = get_firestore()
    if not db:
        return jsonify({'error': 'Firestore unavailable'}), 500
    try:
        snap    = db.collection('courses').stream()
        courses = []
        for doc in snap:
            d = doc.to_dict()
            d['id'] = doc.id
            courses.append(d)
        return jsonify({'courses': courses})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── VIDEO SERVING ─────────────────────────────────────────────────────────────

@app.route('/api/video/list/<exam_id>')
def list_exam_videos(exam_id):
    """Return list of per-question video URLs for a given exam."""
    if not re.match(r'^[\w\-]+$', exam_id):
        return jsonify({'videos': []})
    exam_dir = Path(UPLOAD_DIR) / exam_id
    if not exam_dir.exists():
        return jsonify({'videos': []})
    videos = []
    for i in range(1, 6):
        for ext in ['mp4', 'webm']:
            f = exam_dir / f"q{i}.{ext}"
            if f.exists() and f.stat().st_size > 1000:
                videos.append({
                    'question': i,
                    'url':      f'/api/video/{exam_id}/q{i}.{ext}',
                    'filename': f.name
                })
                break
    return jsonify({'videos': videos})


@app.route('/api/video/<exam_id>/<filename>')
def serve_video(exam_id, filename):
    """Serve a single exam video file from disk."""
    from flask import send_from_directory, abort
    if not re.match(r'^[\w\-]+$', exam_id) or not re.match(r'^[\w\-\.]+$', filename):
        abort(400)
    video_dir = os.path.join(UPLOAD_DIR, exam_id)
    if not os.path.exists(os.path.join(video_dir, filename)):
        abort(404)
    return send_from_directory(video_dir, filename)


# ── QUESTION GENERATION ───────────────────────────────────────────────────────

def extract_text_from_pdf(url):
    """Download a PDF or DOCX from a URL and extract its text."""
    try:
        import requests
        from io import BytesIO

        response = requests.get(url, timeout=30)
        response.raise_for_status()
        content  = response.content

        # Detect file type by magic bytes
        # DOCX/ZIP starts with PK\x03\x04, PDF starts with %PDF
        if content[:4] == b'PK\x03\x04':
            # DOCX file, extract with python-docx
            try:
                import docx
                from io import BytesIO as BIO
                doc  = docx.Document(BIO(content))
                text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
                print(f"   DOCX extracted: {len(text)} chars")
                return text[:6000].strip()
            except ImportError:
                # Fallback: try mammoth
                try:
                    import mammoth
                    result = mammoth.extract_raw_text(BytesIO(content))
                    text   = result.value
                    print(f"   DOCX (mammoth) extracted: {len(text)} chars")
                    return text[:6000].strip()
                except ImportError:
                    print("WARNING: python-docx and mammoth not installed. Run: pip install python-docx mammoth")
                    return ""
        else:
            # PDF file
            from pypdf import PdfReader
            reader = PdfReader(BytesIO(content))
            text   = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            print(f"   PDF extracted: {len(text)} chars")
            return text[:6000].strip()

    except Exception as e:
        print(f"WARNING: Could not extract report text: {e}")
        return ""


def get_group_context(submission_id, student_id):
    context = {
        "course_name":        "Capstone Project",
        "group_name":         "Group",
        "student_name":       "Student",
        "other_members":      [],
        "existing_questions": []
    }
    try:
        db = get_firestore()
        if not db:
            return context

        user_doc = db.collection("users").document(student_id).get()
        if user_doc.exists:
            context["student_name"] = user_doc.to_dict().get("name", "Student")

        sub_doc = db.collection("submissions").document(submission_id).get()
        if not sub_doc.exists:
            return context

        course_id  = sub_doc.to_dict().get("courseId", "")
        course_doc = db.collection("courses").document(course_id).get()
        if course_doc.exists:
            context["course_name"] = course_doc.to_dict().get("name", "Capstone Project")

        groups_snap = db.collection("courses").document(course_id)\
                        .collection("groups").get()
        for group_doc in groups_snap:
            group_data = group_doc.to_dict()
            member_ids = group_data.get("memberIds", [])
            if student_id in member_ids:
                context["group_name"] = group_data.get("name", "Group")
                for uid in member_ids:
                    if uid == student_id:
                        continue
                    m_doc = db.collection("users").document(uid).get()
                    if m_doc.exists:
                        context["other_members"].append(
                            m_doc.to_dict().get("name", uid))
                    exam_id  = f"{submission_id}_{uid}"
                    exam_doc = db.collection("exams").document(exam_id).get()
                    if exam_doc.exists:
                        existing_qs = exam_doc.to_dict().get("questions", [])
                        context["existing_questions"].extend(existing_qs)
                break
    except Exception as e:
        print(f"WARNING: Could not get group context: {e}")
    return context


def generate_rubric_for_student(submission_id, student_id, report_text,
                                 student_name, course_name, group_name,
                                 other_members_str, existing_qs_str,
                                 num_questions=5):
    """
    STAGE 1: Rubric creation.

    Generates a genuine grading rubric from the report: for each question,
    real grading criteria (not just "an answer"), including the specific
    facts/values a correct response must contain. This mirrors how the
    training rubric (rubric.json) was built, a deliberate answer key
    written before any keyword extraction happens, not a raw GPT answer
    mined for keywords after the fact.

    Returns a dict shaped like the training rubric's "questions" section:
        {
          "q1": {
            "question": "...",
            "reference_answer": "...",   # what a correct answer demonstrates
            "required_values": [...]     # specific facts/numbers/terms required
          },
          ...
        }
    """
    prompt = f"""You are a university oral exam examiner building a grading
rubric for an individual contribution assessment.

CONTEXT:
- Course: {course_name}
- Group: {group_name}
- Student being examined: {student_name}
- Other group members: {other_members_str}

PROJECT REPORT CONTENT (extracted text):
\"\"\"
{report_text}
\"\"\"

QUESTIONS ALREADY GIVEN TO OTHER GROUP MEMBERS (do NOT repeat these):
{existing_qs_str}

YOUR TASK:
Build a grading rubric with exactly {num_questions} oral exam questions for
{student_name}, based on SPECIFIC content from the report above.

For each question, write:
1. A question testing {student_name}'s PERSONAL contribution and DEEP
   understanding, not surface-level definitions. It must be unanswerable
   by someone who didn't actually do the work.
2. A "reference_answer": a natural, first-person sample answer showing
   exactly what {student_name} would say if they genuinely did this work
   and answered well. Write it the way a real spoken answer sounds, using
   the specific facts, tools, and terms from the report, not as
   instructions to a grader.
3. "required_values": the specific facts, numbers, technical terms, or
   named components from the report that MUST appear in a correct answer
   (e.g. "PID", "0.82", "transfer function", "settling time"). These are
   the concrete, checkable pieces of evidence a grader would look for.

Use one question of each type:
- Role: "What was your specific contribution to..."
- Technical: "Explain why you chose this specific approach for..."
- Understanding: "How does [specific component from report] work and why?"
- Challenge: "What was the hardest part of your work and how did you solve it?"
- Improvement: "If you had more time, what would you specifically improve?"

RETURN FORMAT: valid JSON only, no markdown, no extra text:
{{
  "questions": [
    {{
      "question": "the question text here",
      "reference_answer": "A first-person sample answer using the specific tools, methods, and terms from this student's own report, written the way a real spoken answer sounds.",
      "required_values": ["specific_term1", "specific_value2", "specific_term3"]
    }}
  ]
}}"""

    client  = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
    message = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = message.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    parsed = json.loads(raw)
    items  = parsed.get("questions", [])[:num_questions]

    rubric = {}
    for i, it in enumerate(items):
        qid = f"q{i+1}"
        rubric[qid] = {
            "question":         it.get("question", ""),
            "reference_answer": it.get("reference_answer", ""),
            "required_values":  it.get("required_values", []),
        }
    return rubric


def extract_scoring_from_rubric(rubric):
    """
    STAGE 2: Extraction.

    Takes the rubric built in Stage 1 and derives the keyword list used by
    feature extraction, the same way the training rubric's shared_extractors
    work: mechanical extraction from an already-written rubric, not from a
    raw model answer.

    Keeps only 5 keywords per question (longest = most specific) to stay
    within the training distribution range of 0-3 keyword hits per window.
    """
    STOPWORDS = {
        "a","an","the","is","are","was","were","be","been","being","have","has",
        "had","do","does","did","will","would","could","should","may","might",
        "shall","can","need","dare","ought","used","to","of","in","for","on",
        "with","at","by","from","as","into","through","about","above","after",
        "before","between","but","if","or","and","not","this","that","these",
        "those","it","its","we","you","i","he","she","they","what","which","who",
        "how","why","when","where","your","our","their","my","his","her","its",
        "so","just","also","then","than","more","most","such","each","both",
        "all","any","few","some","other","own","same","no","nor","too","very",
        "specific","because","student","answer","explain","describe","system",
        "project","work","contribution","approach","method","result","analysis",
    }

    def derive_keywords(reference_answer, required_values):
        # Required values (report-specific facts) always count as keywords,
        # they're the deliberately-chosen checkable evidence from the rubric.
        # required_values can be single words ("PID") or multi-word phrases
        # ("transfer function", "pressure transducer"), both are valid
        # keywords, so this only rejects values that are pure numbers/symbols
        # with no letters at all.
        value_terms = [
            str(v).lower().strip() for v in required_values
            if re.search(r"[a-zA-Z]{3,}", str(v))
        ]

        words = re.findall(r"[a-zA-Z]{4,}", reference_answer.lower())
        seen  = set(value_terms)
        extra = []
        for w in words:
            if w not in STOPWORDS and w not in seen:
                seen.add(w)
                extra.append(w)
        extra.sort(key=len, reverse=True)

        combined = value_terms + extra
        return combined[:5]

    scoring = {}
    for qid, entry in rubric.items():
        scoring[qid] = {
            "question":         entry["question"],
            "expected_answer":  entry["reference_answer"],
            "required_values":  entry["required_values"],
            "keywords":         derive_keywords(
                                     entry["reference_answer"],
                                     entry["required_values"],
                                 ),
        }
    return scoring


def generate_questions_for_student(submission_id, student_id, report_url,
                                    code_url="", slides_url="", num_questions=5):
    """
    Two-stage pipeline:
      Stage 1 (generate_rubric_for_student)   report -> real rubric
      Stage 2 (extract_scoring_from_rubric)   rubric -> keywords/values for
                                               feature extraction

    Both the rubric and the derived scoring are saved, so the rubric itself
    is inspectable as its own artifact (matching how the training rubric.json
    was a standalone, human-readable answer key), not just an intermediate
    value thrown away after keyword extraction.
    """
    print(f"Generating rubric and questions for student {student_id}...")

    # Report is required; without it we can't generate meaningful questions
    if not report_url:
        print(f"ERROR: No report URL provided for {student_id}. Aborting.")
        return

    print("   Extracting report text...")
    report_text = extract_text_from_pdf(report_url)

    if not report_text:
        print(f"ERROR: Could not extract report text for {student_id}. Aborting.")
        return

    print("   Fetching group context...")
    ctx = get_group_context(submission_id, student_id)

    student_name  = ctx["student_name"]
    course_name   = ctx["course_name"]
    group_name    = ctx["group_name"]
    other_members = ctx["other_members"]
    existing_qs   = ctx["existing_questions"]

    other_members_str = ", ".join(other_members) if other_members else "none yet"
    existing_qs_str   = "\n".join(f"- {q}" for q in existing_qs) \
                        if existing_qs else "None yet"

    try:
        print("   Stage 1: generating rubric...")
        rubric = generate_rubric_for_student(
            submission_id, student_id, report_text,
            student_name, course_name, group_name,
            other_members_str, existing_qs_str, num_questions,
        )

        print("   Stage 2: extracting keywords from rubric...")
        questions_data = extract_scoring_from_rubric(rubric)

        questions        = [questions_data[f"q{i+1}"]["question"]        for i in range(len(questions_data))]
        expected_answers = [questions_data[f"q{i+1}"]["expected_answer"] for i in range(len(questions_data))]

        print(f"Generated {len(questions)} questions for {student_name}")
        for i, q in enumerate(questions):
            print(f"   Q{i+1}: {q[:80]}...")

        exam_id  = f"{submission_id}_{student_id}"
        exam_dir = Path(UPLOAD_DIR) / exam_id
        exam_dir.mkdir(parents=True, exist_ok=True)

        # Save the rubric itself as its own artifact, inspectable,
        # matching training rubric.json's role as a standalone answer key.
        rubric_path = exam_dir / "rubric.json"
        with open(str(rubric_path), "w", encoding="utf-8") as f:
            json.dump(rubric, f, indent=2)
        print(f"rubric.json written: {rubric_path}")

        # Save the extracted scoring data; this is what feature
        # extraction actually reads (keywords, required_values).
        questions_json_path = exam_dir / "questions.json"
        with open(str(questions_json_path), "w", encoding="utf-8") as f:
            json.dump(questions_data, f, indent=2)
        print(f"questions.json written: {questions_json_path}")

        db = get_firestore()
        if db:
            exam_ref = db.collection('exams').document(exam_id)
            exam_ref.set({
                'submissionId':    submission_id,
                'studentId':       student_id,
                'studentName':     student_name,
                'courseName':      course_name,
                'groupName':       group_name,
                'questions':       questions,
                'expectedAnswers': expected_answers,
                'keywords_q1':     questions_data.get('q1',{}).get('keywords',[]),
                'keywords_q2':     questions_data.get('q2',{}).get('keywords',[]),
                'keywords_q3':     questions_data.get('q3',{}).get('keywords',[]),
                'keywords_q4':     questions_data.get('q4',{}).get('keywords',[]),
                'keywords_q5':     questions_data.get('q5',{}).get('keywords',[]),
                'answerTimeSecs':  15,  # matches TRAINING_MATCH_DURATION_SEC in audio_features.py,
                                         # giving longer than what's actually scored was misleading students
                'status':          'ready',
                'createdAt':       firestore.SERVER_TIMESTAMP,
                'startedAt':       None,
                'completedAt':     None,
            })
            print(f"Exam sealed in Firestore: {exam_id}")
        else:
            print("WARNING: Firestore not available.")

    except json.JSONDecodeError as e:
        print(f"ERROR: GPT returned invalid JSON: {e}")
    except Exception as e:
        print(f"ERROR: Question generation failed for {student_id}: {e}")
        traceback.print_exc()


@app.route('/api/questions/generate', methods=['POST'])
def api_generate_questions():
    """
    Called by student_portal.html after all files are uploaded.
    V2: report_url is now required.
    """
    data = request.json or {}

    submission_id = data.get('submissionId')
    student_id    = data.get('studentId')
    report_url    = data.get('reportUrl', '')
    code_url      = data.get('codeUrl', '')
    slides_url    = data.get('slidesUrl', '')
    num_questions = int(data.get('numQuestions', 5))

    if not submission_id or not student_id:
        return jsonify({'error': 'submissionId and studentId are required'}), 400

    # V2: Report is required
    if not report_url:
        return jsonify({
            'error': 'Report must be uploaded before generating exam questions.'
        }), 400

    thread = threading.Thread(
        target=generate_questions_for_student,
        args=(submission_id, student_id, report_url,
              code_url, slides_url, num_questions),
        daemon=True
    )
    thread.start()

    return jsonify({
        'success': True,
        'message': 'Question generation started. Exam will be ready in ~30 seconds.',
        'examId':  f"{submission_id}_{student_id}"
    })


# ── VIDEO UPLOAD & FEATURE EXTRACTION ────────────────────────────────────────

@app.route('/api/video/upload', methods=['POST'])
def api_video_upload():
    """
    V2: Called by viva_exam.html after each question answer is recorded.

    Receives the video blob, saves it, fetches the expected answer for that
    question from Firestore, runs feature_extractor, and returns the
    windowed CSV path for use in the inference pipeline.

    Form fields:
        video     : video file blob (webm)
        examId    : exam document ID
        questionIndex : 0-based index of the question (0 = Q1, 4 = Q5)
    """
    exam_id       = request.form.get('examId')
    q_index       = int(request.form.get('questionIndex', 0))
    video_file    = request.files.get('video')

    if not exam_id or not video_file:
        return jsonify({'error': 'examId and video are required'}), 400

    try:
        # Save video to disk
        exam_dir   = Path(UPLOAD_DIR) / exam_id
        exam_dir.mkdir(parents=True, exist_ok=True)

        # Fetch the full rubric entry for this question from questions.json
        # (written by generate_questions_for_student's two-stage rubric
        # pipeline at question-generation time), includes expected_answer,
        # keywords, and required_values, all needed for correctness/keyword
        # scoring in feature_extractor.
        question_id  = f"q{q_index + 1}"
        rubric_entry = {}

        questions_json_path = exam_dir / "questions.json"
        if questions_json_path.exists():
            try:
                with open(str(questions_json_path), "r", encoding="utf-8") as f:
                    questions_data = json.load(f)
                rubric_entry = questions_data.get(question_id, {})
            except Exception as e:
                print(f"WARNING: Could not read questions.json: {e}")

        print(f"Rubric entry loaded: {bool(rubric_entry)}")

        # Save raw webm; extraction is deferred until all 5 questions are
        # uploaded (see run_all_extractions_sequential / delayed_pipeline),
        # so the 5 CPU-heavy extraction jobs (Whisper + MediaPipe + Librosa)
        # run one at a time instead of competing for the same CPU cores.
        # Running them in parallel was observed to slow MediaPipe down by
        # 10-20x on this machine (250-500 frames/s solo vs. 11-22 frames/s
        # under contention), which more than erased any head-start benefit.
        webm_path  = exam_dir / f"q{q_index + 1}.webm"
        video_file.save(str(webm_path))
        print(f"Webm saved: {webm_path.name}  ({webm_path.stat().st_size // 1024}KB)")

        csv_path = exam_dir / f"q{q_index + 1}_windowed.csv"

        return jsonify({
            'success':     True,
            'csvPath':     str(csv_path),
            'questionId':  question_id,
            'message':     f'Video received for Q{q_index + 1}. Will process after exam completes.'
        })

    except Exception as e:
        print(f"ERROR: Video upload failed: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ── PIPELINE ──────────────────────────────────────────────────────────────────

def extract_one_question(exam_dir, q_index, rubric_entry):
    """
    Runs the full webm-to-features pipeline for one question. Called
    sequentially, one question at a time, after all 5 videos are uploaded
    (see run_all_extractions_sequential); extraction jobs are CPU-heavy
    (Whisper + MediaPipe + Librosa) and were observed to slow each other
    down by 10-20x when run in parallel on this machine.
    """
    import time, subprocess

    q_index_1     = q_index + 1
    question_id   = f"q{q_index_1}"
    webm_path     = exam_dir / f"q{q_index_1}.webm"
    csv_path      = exam_dir / f"q{q_index_1}_windowed.csv"

    try:
        # Step 1: wait until the file size stops changing, confirming
        # the browser/OS has fully finished writing it.
        last_size = -1
        stable_checks = 0
        for _ in range(20):  # up to ~20s
            current_size = webm_path.stat().st_size if webm_path.exists() else 0
            if current_size > 0 and current_size == last_size:
                stable_checks += 1
                if stable_checks >= 2:
                    break
            else:
                stable_checks = 0
            last_size = current_size
            time.sleep(1)

        # Step 2: verify file is big enough to be real video
        size = webm_path.stat().st_size if webm_path.exists() else 0
        if size < 1000:
            print(f"WARNING: Video too small ({size}B), skipping Q{q_index_1}")
            return

        # Step 3: convert webm to mp4 (fixes EBML header issues from MediaRecorder)
        mp4_path = exam_dir / f"q{q_index_1}.mp4"
        cmd = [
            "ffmpeg", "-y", "-loglevel", "warning",
            "-f", "webm",
            "-i", str(webm_path),
            "-c:v", "libx264", "-c:a", "aac",
            "-movflags", "+faststart",
            str(mp4_path)
        ]
        subprocess.run(cmd, capture_output=True, text=True)

        if mp4_path.exists() and mp4_path.stat().st_size > 1000:
            process_path = mp4_path
            print(f"Converted to mp4: {mp4_path.name}  ({mp4_path.stat().st_size // 1024}KB)")
        else:
            # mp4 conversion failed, try again with error-tolerant flags.
            # The last question's webm can arrive with a malformed/
            # incomplete container index if the browser's MediaRecorder
            # stops right as the page navigates away, a known webm
            # quirk, not a corrupted recording.
            print(f"WARNING: libx264 failed, retrying with error-tolerant flags...")
            cmd2 = [
                "ffmpeg", "-y", "-loglevel", "warning",
                "-fflags", "+genpts+igndts",
                "-analyzeduration", "20M", "-probesize", "20M",
                "-f", "webm",
                "-i", str(webm_path),
                "-c:v", "libx264", "-c:a", "aac",
                "-movflags", "+faststart",
                str(mp4_path)
            ]
            subprocess.run(cmd2, capture_output=True, text=True)
            if mp4_path.exists() and mp4_path.stat().st_size > 1000:
                process_path = mp4_path
                print(f"Recovered and converted to mp4: {mp4_path.name}")
            else:
                # Last resort: try re-reading the webm one more time
                # after a short pause. If it's still unreadable, fail
                # cleanly rather than crashing inside feature_extractor.
                print(f"WARNING: mp4 conversion failed, retrying webm read after pause...")
                time.sleep(3)

                import cv2 as _cv2
                test_cap = _cv2.VideoCapture(str(webm_path))
                webm_readable = test_cap.isOpened()
                if webm_readable:
                    frame_ok, _ = test_cap.read()
                    webm_readable = bool(frame_ok)
                test_cap.release()

                if not webm_readable:
                    print(
                        f"ERROR: q{q_index_1} video is unreadable by both "
                        f"ffmpeg and OpenCV, skipping extraction for this "
                        f"question. It will show as missing in the "
                        f"completeness check."
                    )
                    return

                process_path = webm_path
                print(f"WARNING: Using webm directly: {webm_path.name}")

        # Step 4: run windowed feature extraction
        from feature_extractor import extract_windowed
        _, answer_row = extract_windowed(
            video_path_str = str(process_path),
            whisper_model  = "small.en",
            output_dir     = str(exam_dir),
            window_sec     = 3.0,
            question_id    = question_id,
            rubric_entry   = rubric_entry,
        )

        answer_row_path = exam_dir / f"{question_id}_answer_row.json"
        with open(str(answer_row_path), "w", encoding="utf-8") as f:
            json.dump(answer_row, f, default=str)

        print(f"Extraction done: {csv_path.name}")

        # Step 5: fix_broken_videos style re-encode fallback if CSV is empty/missing
        if not csv_path.exists() or csv_path.stat().st_size < 100:
            print(f"WARNING: CSV empty, trying re-encode fix for {process_path.name}")
            fixed_path = exam_dir / f"q{q_index_1}_fixed.mp4"
            fix_cmd = [
                "ffmpeg", "-y", "-loglevel", "warning",
                "-i", str(process_path),
                "-vcodec", "libx264", "-acodec", "aac",
                "-strict", "experimental",
                str(fixed_path)
            ]
            subprocess.run(fix_cmd, capture_output=True, text=True)
            if fixed_path.exists() and fixed_path.stat().st_size > 1000:
                _, answer_row = extract_windowed(
                    video_path_str = str(fixed_path),
                    whisper_model  = "small.en",
                    output_dir     = str(exam_dir),
                    window_sec     = 3.0,
                    question_id    = question_id,
                    rubric_entry   = rubric_entry,
                )
                answer_row_path = exam_dir / f"{question_id}_answer_row.json"
                with open(str(answer_row_path), "w", encoding="utf-8") as f:
                    json.dump(answer_row, f, default=str)
                print(f"Fixed extraction done: {csv_path.name}")

    except Exception as e:
        print(f"ERROR: Extraction failed for Q{q_index_1}: {e}")
        traceback.print_exc()


def run_all_extractions_sequential(exam_id):
    """
    Runs extract_one_question for all 5 questions, one at a time, in order.
    Called once after all 5 videos are uploaded (see api_run_pipeline),
    instead of starting extraction per-question during the exam. Keeps
    CPU-heavy jobs from competing with each other.
    """
    exam_dir = Path(UPLOAD_DIR) / exam_id

    questions_json_path = exam_dir / "questions.json"
    questions_data = {}
    if questions_json_path.exists():
        try:
            with open(str(questions_json_path), "r", encoding="utf-8") as f:
                questions_data = json.load(f)
        except Exception as e:
            print(f"WARNING: Could not read questions.json: {e}")

    for q_index in range(5):
        question_id  = f"q{q_index + 1}"
        rubric_entry = questions_data.get(question_id, {})

        csv_path        = exam_dir / f"{question_id}_windowed.csv"
        answer_row_path = exam_dir / f"{question_id}_answer_row.json"

        if csv_path.exists() and answer_row_path.exists():
            print(f"  {question_id.upper()} already extracted, skipping.")
            continue

        webm_path = exam_dir / f"{question_id}.webm"
        if not webm_path.exists():
            print(f"  WARNING: No video found for {question_id.upper()}, skipping.")
            continue

        print(f"  Extracting {question_id.upper()}...")
        extract_one_question(exam_dir, q_index, rubric_entry)


def run_pipeline_for_exam(exam_id, student_id):
    """
    V3: Called after all videos have been uploaded and processed.
    Looks for windowed CSVs + answer-row JSONs in data/exam_videos/{exam_id}/
    and runs the GRU+RF inference pipeline on them.

    A session with fewer than 5 complete answers is never sent to
    inference. GRU requires the full 5-question sequence, and predicting
    on a partial session would be meaningless. Such sessions are saved as
    "pending_video" instead, same as before.
    """
    print(f"Running pipeline for exam {exam_id}...")
    try:
        db = get_firestore()
        if not db:
            return

        result_ref = db.collection("results").document(exam_id)

        # Find windowed CSVs + answer-row JSONs for this exam
        exam_dir = Path(UPLOAD_DIR) / exam_id
        csv_paths = []
        answer_rows = []
        quality_flags = []
        missing = []

        for q_idx in range(1, 6):   # Q1 to Q5
            question_id = f"q{q_idx}"
            csv_file = exam_dir / f"{question_id}_windowed.csv"
            answer_row_file = exam_dir / f"{question_id}_answer_row.json"

            if csv_file.exists() and answer_row_file.exists():
                csv_paths.append(str(csv_file))
                with open(str(answer_row_file), "r", encoding="utf-8") as f:
                    answer_row = json.load(f)
                answer_rows.append(answer_row)
                quality_flags.append(bool(answer_row.get("needs_live_review", False)))
            else:
                missing.append(question_id)
                print(f"  WARNING: Missing data for {question_id.upper()}")

        # Completeness gate: inference cannot run on a partial session
        if len(csv_paths) < 5:
            result_ref.set({
                "studentId":      student_id,
                "submissionId":   exam_id.split("_")[0] if "_" in exam_id else exam_id,
                "prediction":     None,
                "authenticProb":  None,
                "confidence":     None,
                "freeRiderFlag":  False,
                "pipelineStatus": "pending_video",
                "missingQuestions": missing,
                "createdAt":      firestore.SERVER_TIMESTAMP,
            })
            print(
                f"WARNING: Only {len(csv_paths)}/5 answers ready for "
                f"{exam_id}: saved pending placeholder, missing: {missing}"
            )
            return

        print(f"  All 5 answers ready, running inference...")

        from inference import VivaAIPredictor
        predictor = VivaAIPredictor()
        result = predictor.predict_student(
            csv_paths, answer_rows=answer_rows, quality_flags=quality_flags
        )

        # Read per-question transcripts saved by feature_extractor
        transcripts_by_q = {}
        combined_transcript = ""
        for q_idx in range(1, 6):
            txt_file = exam_dir / f"q{q_idx}_transcript.txt"
            if txt_file.exists():
                try:
                    raw = txt_file.read_text(encoding="utf-8")
                    marker = "--- FULL TRANSCRIPT ---"
                    plain  = raw.split(marker)[1].split("--- SEGMENTS ---")[0].strip() if marker in raw else raw.strip()
                    if plain:
                        transcripts_by_q[f"q{q_idx}"] = plain
                        combined_transcript += f"[Q{q_idx}] {plain}\n\n"
                except Exception as e:
                    print(f"WARNING: Could not read transcript Q{q_idx}: {e}")

        result_ref.set({
            "studentId":          student_id,
            "submissionId":       exam_id.split("_")[0] if "_" in exam_id else exam_id,
            # Primary decision: GRU's own call, per the two-model
            # architecture (GRU decides, RF is a disagreement check only).
            # This is what review_page.html's banner/pill show. It's
            # always GRU's own call, independent of freeRiderFlag below.
            "prediction":         result["gru_prediction"],
            "authenticProb":      result["gru_probability"],
            "confidence":         result["confidence"],
            # "Flagged" (shown on the dashboard's Flagged Students page)
            # means a RESOLVED, confident free-rider call. GRU says
            # free-rider AND RF agrees AND it's not a gray-zone case.
            # A GRU-alone free-rider lean that still needs manual review
            # should NOT show up here; it belongs in the review queue
            # until an instructor confirms it (via Confirm Flag / Final
            # Decision, which sets this field directly in review_page.html).
            "freeRiderFlag":      (
                result["gru_prediction"] == "free_rider"
                and not result["mandatory_review"]
            ),
            # Second-model comparison, used only for the review trigger.
            "rfProbability":      result["rf_probability"],
            "rfPrediction":       result["rf_prediction"],
            "modelsAgree":        result["models_agree"],
            # Three-trigger mandatory review policy.
            "modelDisagreement":  result["model_disagreement"],
            "grayZone":           result["gray_zone"],
            "qualityFlag":        result["quality_flag"],
            "mandatoryReview":    result["mandatory_review"],
            "featuresUsed":       result["features_used"],
            "pipelineStatus":     "complete",
            "transcript":         combined_transcript.strip(),
            "transcriptByQ":      transcripts_by_q,
            "createdAt":          firestore.SERVER_TIMESTAMP,
        })
        print(
            f"Result saved for {exam_id}: GRU={result['gru_prediction']} "
            f"(prob={result['gru_probability']:.3f}) | "
            f"RF={result['rf_prediction']} (prob={result['rf_probability']:.3f}) | "
            f"review={'YES' if result['mandatory_review'] else 'no'}"
        )

    except Exception as e:
        print(f"ERROR: Pipeline failed for {exam_id}: {e}")
        traceback.print_exc()
        try:
            db = get_firestore()
            if db:
                db.collection("results").document(exam_id).set({
                    "studentId":      student_id,
                    "submissionId":   exam_id.split("_")[0] if "_" in exam_id else exam_id,
                    "prediction":     None,
                    "authenticProb":  None,
                    "confidence":     None,
                    "freeRiderFlag":  False,
                    "pipelineStatus": "error",
                    "errorMessage":   str(e),
                    "createdAt":      firestore.SERVER_TIMESTAMP,
                }, merge=True)
        except Exception as save_err:
            print(f"ERROR: Could not save error status for {exam_id}: {save_err}")


@app.route('/api/pipeline/run', methods=['POST'])
def api_run_pipeline():
    """
    Called by viva_exam.html when exam completes.
    V2: Waits for extraction to finish then runs inference.
    """
    data       = request.json or {}
    exam_id    = data.get('examId')
    student_id = data.get('studentId')

    if not exam_id or not student_id:
        return jsonify({'error': 'examId and studentId are required'}), 400

    def delayed_pipeline():
        # Run all 5 extractions one at a time, in order. No per-question
        # background threads anymore, so there's nothing to "wait for"
        # racing in parallel. This call blocks until all questions that
        # have a video have been processed.
        print(f"  Running sequential extraction for {exam_id}...")
        run_all_extractions_sequential(exam_id)
        run_pipeline_for_exam(exam_id, student_id)

    thread = threading.Thread(target=delayed_pipeline, daemon=True)
    thread.start()

    return jsonify({'success': True, 'message': 'Pipeline started in background'})


@app.route('/api/pipeline/result/<exam_id>', methods=['GET'])
def api_pipeline_result(exam_id):
    """Check if pipeline result is ready for an exam."""
    db = get_firestore()
    if not db:
        return jsonify({'status': 'unavailable'})
    doc = db.collection('results').document(exam_id).get()
    if doc.exists:
        return jsonify({'status': 'ready', 'result': doc.to_dict()})
    return jsonify({'status': 'pending'})


# ── EXAM STATUS ───────────────────────────────────────────────────────────────

@app.route('/api/exam/status/<exam_id>', methods=['GET'])
def api_exam_status(exam_id):
    db = get_firestore()
    if not db:
        return jsonify({'status': 'unavailable'})
    doc = db.collection('exams').document(exam_id).get()
    if doc.exists:
        return jsonify({'status': doc.to_dict().get('status', 'pending')})
    return jsonify({'status': 'not_found'})


# ── RUN ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("VivaAI v2 starting...")
    print(f"   OPENAI_API_KEY: {'set' if os.environ.get('OPENAI_API_KEY') else 'ERROR: missing'}")
    print(f"   FIREBASE_CREDENTIALS: {'found' if os.path.exists(os.environ.get('FIREBASE_CREDENTIALS','serviceAccountKey.json')) else 'WARNING: not found'}")
    app.run(debug=True, port=5000)
