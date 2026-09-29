/**
 * VivaAI — Firebase Backend
 * ==========================
 * Drop this file next to your HTML pages.
 * Import it in every page using:
 *   <script type="module" src="firebase.js"></script>
 *
 * Or if using Flask static folder:
 *   <script type="module" src="/static/js/firebase.js"></script>
 */

import { initializeApp } from "https://www.gstatic.com/firebasejs/10.12.0/firebase-app.js";
import {
  getAuth,
  signInWithEmailAndPassword,
  createUserWithEmailAndPassword,
  signOut,
  onAuthStateChanged
} from "https://www.gstatic.com/firebasejs/10.12.0/firebase-auth.js";
import {
  getFirestore,
  doc,
  setDoc,
  getDoc,
  getDocs,
  addDoc,
  updateDoc,
  deleteDoc,
  collection,
  query,
  where,
  orderBy,
  serverTimestamp
} from "https://www.gstatic.com/firebasejs/10.12.0/firebase-firestore.js";
import {
  getStorage,
  ref,
  uploadBytes,
  getDownloadURL
} from "https://www.gstatic.com/firebasejs/10.12.0/firebase-storage.js";

// ── CONFIG ────────────────────────────────────────────────────────────────────

const firebaseConfig = {
  apiKey: "AIzaSyBH4PDuzlqtg0kq-zBZID18p9-v0BoqKfM",
  authDomain: "vivaai-9fe60.firebaseapp.com",
  projectId: "vivaai-9fe60",
  storageBucket: "vivaai-9fe60.firebasestorage.app",
  messagingSenderId: "593955807088",
  appId: "1:593955807088:web:b25228dde849dc42d179b5"
};

const app  = initializeApp(firebaseConfig);
const auth = getAuth(app);
const db   = getFirestore(app);
const storage = getStorage(app);

// ── AUTH FUNCTIONS ────────────────────────────────────────────────────────────

/**
 * Sign in existing user.
 * Returns: { uid, email, role, name }
 */
export async function signIn(email, password) {
  const cred = await signInWithEmailAndPassword(auth, email, password);
  const userDoc = await getDoc(doc(db, "users", cred.user.uid));
  if (!userDoc.exists()) throw new Error("User profile not found.");
  return { uid: cred.user.uid, ...userDoc.data() };
}

/**
 * Register a new student.
 * role is always "student" for self-signup.
 */
export async function registerStudent(name, studentId, email, password) {
  const cred = await createUserWithEmailAndPassword(auth, email, password);
  await setDoc(doc(db, "users", cred.user.uid), {
    name,
    studentId,
    email,
    role: "student",
    createdAt: serverTimestamp()
  });
  return cred.user.uid;
}

/**
 * Register an instructor (call this manually or from admin setup).
 */
export async function registerInstructor(name, email, password) {
  const cred = await createUserWithEmailAndPassword(auth, email, password);
  await setDoc(doc(db, "users", cred.user.uid), {
    name,
    email,
    role: "instructor",
    createdAt: serverTimestamp()
  });
  return cred.user.uid;
}

/**
 * Sign out current user.
 */
export async function logOut() {
  await signOut(auth);
  window.location.href = "/login";
}

/**
 * Get the currently logged-in user's profile from Firestore.
 * Returns null if not logged in.
 */
export function getCurrentUser() {
  return new Promise((resolve) => {
    onAuthStateChanged(auth, async (user) => {
      if (!user) return resolve(null);
      const snap = await getDoc(doc(db, "users", user.uid));
      resolve(snap.exists() ? { uid: user.uid, ...snap.data() } : null);
    });
  });
}

/**
 * Redirect to login if not authenticated.
 * Redirect to correct dashboard based on role.
 * Usage: await requireAuth("student")  or  await requireAuth("instructor")
 */
export async function requireAuth(expectedRole) {
  const user = await getCurrentUser();
  if (!user) {
    window.location.href = "/login";
    return null;
  }
  if (expectedRole && user.role !== expectedRole) {
    // Wrong role — send to correct page
    window.location.href = user.role === "instructor" ? "/instructor" : "/student";
    return null;
  }
  return user;
}

// ── COURSES ───────────────────────────────────────────────────────────────────

/**
 * Create a new course (instructor only).
 */
export async function createCourse(name, code) {
  const user = auth.currentUser;
  const ref = await addDoc(collection(db, "courses"), {
    name,
    code,
    instructorId: user.uid,
    createdAt: serverTimestamp()
  });
  return ref.id;
}

/**
 * Get all courses for instructor.
 */
export async function getInstructorCourses() {
  const user = auth.currentUser;
  const q = query(collection(db, "courses"), where("instructorId", "==", user.uid));
  const snap = await getDocs(q);
  return snap.docs.map(d => ({ id: d.id, ...d.data() }));
}

/**
 * Add a group to a course.
 */
export async function addGroup(courseId, groupName, maxSize = 5) {
  const ref = await addDoc(collection(db, "courses", courseId, "groups"), {
    name: groupName,
    memberIds: [],
    maxSize,
    createdAt: serverTimestamp()
  });
  return ref.id;
}

/**
 * Get groups for a course.
 */
export async function getCourseGroups(courseId) {
  const snap = await getDocs(collection(db, "courses", courseId, "groups"));
  return snap.docs.map(d => ({ id: d.id, ...d.data() }));
}

/**
 * Student joins a group.
 */
export async function joinGroup(courseId, groupId) {
  const user = auth.currentUser;
  const groupRef = doc(db, "courses", courseId, "groups", groupId);
  const groupSnap = await getDoc(groupRef);
  if (!groupSnap.exists()) throw new Error("Group not found.");
  const { memberIds, maxSize } = groupSnap.data();
  if (memberIds.includes(user.uid)) throw new Error("Already in this group.");
  if (memberIds.length >= maxSize) throw new Error("Group is full.");
  await updateDoc(groupRef, { memberIds: [...memberIds, user.uid] });
}

// ── SUBMISSIONS ───────────────────────────────────────────────────────────────

/**
 * Create a submission slot (instructor creates, students fill it).
 */
export async function createSubmission(courseId, groupId, title) {
  const ref = await addDoc(collection(db, "submissions"), {
    courseId,
    groupId,
    title,
    status: "pending",       // pending | submitted | examined
    reportUrl: null,
    createdAt: serverTimestamp()
  });
  return ref.id;
}

/**
 * Upload student report PDF to Storage, save URL to submission.
 */
export async function uploadReport(submissionId, file) {
  const user = auth.currentUser;
  const path = `reports/${submissionId}/${user.uid}_${file.name}`;
  const storageRef = ref(storage, path);
  await uploadBytes(storageRef, file);
  const url = await getDownloadURL(storageRef);
  await updateDoc(doc(db, "submissions", submissionId), {
    reportUrl: url,
    submittedBy: user.uid,
    submittedAt: serverTimestamp(),
    status: "submitted"
  });
  return url;
}

/**
 * Get submission by ID.
 */
export async function getSubmission(submissionId) {
  const snap = await getDoc(doc(db, "submissions", submissionId));
  return snap.exists() ? { id: snap.id, ...snap.data() } : null;
}

// ── EXAMS & QUESTIONS ─────────────────────────────────────────────────────────

/**
 * Seal generated questions into Firestore for a student's exam.
 * Questions are stored encrypted (hidden) until exam starts.
 * Called by backend after Claude generates questions.
 */
export async function sealExam(submissionId, studentId, questions) {
  const examId = `${submissionId}_${studentId}`;
  await setDoc(doc(db, "exams", examId), {
    submissionId,
    studentId,
    questions,          // array of question strings
    status: "ready",    // ready | in_progress | completed
    createdAt: serverTimestamp(),
    startedAt: null,
    completedAt: null
  });
  return examId;
}

/**
 * Load exam for student — only works if exam status is "ready" or "in_progress".
 */
export async function loadExam(examId) {
  const snap = await getDoc(doc(db, "exams", examId));
  if (!snap.exists()) throw new Error("Exam not found.");
  return { id: snap.id, ...snap.data() };
}

/**
 * Mark exam as started.
 */
export async function startExam(examId) {
  await updateDoc(doc(db, "exams", examId), {
    status: "in_progress",
    startedAt: serverTimestamp()
  });
}

/**
 * Mark exam as completed and save results.
 */
export async function completeExam(examId, results) {
  await updateDoc(doc(db, "exams", examId), {
    status: "completed",
    completedAt: serverTimestamp(),
    results
  });
}

// ── RESULTS ───────────────────────────────────────────────────────────────────

/**
 * Save pipeline results (7 features + contribution score) for a student.
 */
export async function savePipelineResult(studentId, submissionId, features) {
  await setDoc(doc(db, "results", `${submissionId}_${studentId}`), {
    studentId,
    submissionId,
    features,           // { blink_rate, pitch_variation, speech_energy, ... }
    contributionScore: features.contribution_score ?? null,
    freeRiderFlag: features.free_rider ?? false,
    createdAt: serverTimestamp()
  });
}

/**
 * Get all results for instructor review.
 */
export async function getAllResults(courseId) {
  const q = query(collection(db, "results"), where("courseId", "==", courseId));
  const snap = await getDocs(q);
  return snap.docs.map(d => ({ id: d.id, ...d.data() }));
}

/**
 * Save instructor's final review and grade.
 */
export async function saveReview(studentId, submissionId, grade, notes) {
  await updateDoc(doc(db, "results", `${submissionId}_${studentId}`), {
    grade,
    reviewNotes: notes,
    reviewedAt: serverTimestamp()
  });
}

// ── EXPORT DB & AUTH (for advanced use) ───────────────────────────────────────
export { auth, db, storage };
