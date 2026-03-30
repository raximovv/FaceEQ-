import cv2
import mediapipe as mp
import numpy as np
import sounddevice as sd
import soundfile as sf
import threading
import sys
import os
 
def find_track():
    for name in ["track.wav", "track.mp3", "track.ogg", "track.flac"]:
        if os.path.exists(name):
            return name
    return None
 
track_file = find_track()
if track_file is None:
    print("ERROR: No track file found. Place track.wav or track.mp3 here.")
    sys.exit(1)
 
print(f"Loading: {track_file} ...")
try:
    audio_data, sample_rate = sf.read(track_file, dtype='float32', always_2d=True)
except Exception as e:
    print(f"ERROR loading audio: {e}")
    print("Try: ffmpeg -i track.mp3 track.wav")
    sys.exit(1)
 
if audio_data.shape[1] == 1:
    audio_data = np.repeat(audio_data, 2, axis=1)
 
TOTAL_SAMPLES = len(audio_data)
TRACK_DURATION = TOTAL_SAMPLES / sample_rate
print(f"Track loaded: {TRACK_DURATION:.1f}s, {sample_rate}Hz, {audio_data.shape[1]}ch")
 
 
class AudioEngine:
    def __init__(self, data, sr):
        self.data = data
        self.sr = sr
        self.total = len(data)
        self._lock = threading.Lock()
        self._pos = 0
        self._speed = 1.0
        self._playing = False
        self._stream = None
 
    @property
    def position_sec(self):
        with self._lock:
            return self._pos / self.sr
 
    @position_sec.setter
    def position_sec(self, secs):
        with self._lock:
            self._pos = max(0, min(self.total - 1, int(secs * self.sr)))
 
    @property
    def speed(self):
        with self._lock:
            return self._speed
 
    @speed.setter
    def speed(self, val):
        with self._lock:
            self._speed = max(0.25, min(3.0, val))
 
    @property
    def is_playing(self):
        with self._lock:
            return self._playing
 
    def _callback(self, outdata, frames, time_info, status):
        with self._lock:
            if not self._playing:
                outdata[:] = 0
                return
            spd = self._speed
            out = np.zeros((frames, 2), dtype=np.float32)
            pos = float(self._pos)
            for i in range(frames):
                idx = int(pos)
                if idx >= self.total - 1:
                    out[i] = 0
                    self._playing = False
                    self._pos = 0
                    break
                frac = pos - idx
                out[i] = self.data[idx] * (1 - frac) + self.data[idx + 1] * frac
                pos += spd
            self._pos = pos
            outdata[:] = out
 
    def start(self):
        self._stream = sd.OutputStream(
            samplerate=self.sr,
            channels=2,
            dtype='float32',
            blocksize=1024,
            callback=self._callback
        )
        self._stream.start()
 
    def play(self):
        with self._lock:
            self._playing = True
 
    def toggle(self):
        with self._lock:
            self._playing = not self._playing
        return self._playing
 
    def reset(self):
        with self._lock:
            self._pos = 0
            self._playing = False
 
    def stop(self):
        if self._stream:
            self._stream.stop()
            self._stream.close()
 
 
engine = AudioEngine(audio_data, sample_rate)
engine.start()
engine.play()
 
NOSE_TIP = 1
CHIN = 152
LEFT_EYE = 33
RIGHT_EYE = 263
FOREHEAD = 10
 
def estimate_head_pose(landmarks, w, h):
    nose      = landmarks[NOSE_TIP]
    left_eye  = landmarks[LEFT_EYE]
    right_eye = landmarks[RIGHT_EYE]
    forehead  = landmarks[FOREHEAD]
    chin      = landmarks[CHIN]
 
    eye_mid_x    = (left_eye.x + right_eye.x) / 2
    eye_distance = abs(right_eye.x - left_eye.x)
    yaw = ((nose.x - eye_mid_x) / eye_distance) * 60 if eye_distance > 0 else 0.0
 
    nose_to_chin     = chin.y - nose.y
    forehead_to_nose = nose.y - forehead.y
    pitch = (nose_to_chin / forehead_to_nose - 1.0) * 40 if forehead_to_nose > 0 else 0.0
 
    return yaw, pitch
 
YAW_DEAD_ZONE   = 5.0
PITCH_DEAD_ZONE = 5.0
SCRUB_SPEED     = 0.3
MIN_SPEED       = 0.4
MAX_SPEED       = 2.5
SMOOTHING       = 0.25
 
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("ERROR: No webcam found.")
    engine.stop()
    sys.exit(1)
 
ret, test_frame = cap.read()
if not ret:
    print("ERROR: Can't read from webcam.")
    engine.stop()
    sys.exit(1)
 
FRAME_H, FRAME_W = test_frame.shape[:2]
 
mp_face   = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
 
smooth_yaw    = 0.0
smooth_pitch  = 0.0
current_yaw   = 0.0
current_pitch = 0.0
 
def draw_gauge(frame, cx, cy, radius, yaw, pitch):
    cv2.circle(frame, (cx, cy), radius, (30, 30, 30), -1)
    cv2.circle(frame, (cx, cy), radius, (120, 120, 120), 2)
    cv2.line(frame, (cx - radius, cy), (cx + radius, cy), (60, 60, 60), 1)
    cv2.line(frame, (cx, cy - radius), (cx, cy + radius), (60, 60, 60), 1)
    dz_r = int(radius * (YAW_DEAD_ZONE / 30.0))
    cv2.circle(frame, (cx, cy), dz_r, (60, 60, 60), 1)
    dot_x   = int(cx + np.clip(yaw   / 30.0, -1, 1) * radius)
    dot_y   = int(cy - np.clip(pitch / 20.0, -1, 1) * radius)
    in_dead = abs(yaw) < YAW_DEAD_ZONE and abs(pitch) < PITCH_DEAD_ZONE
    color   = (100, 100, 100) if in_dead else (0, 220, 255)
    cv2.circle(frame, (dot_x, dot_y), 7, color, -1)
    cv2.putText(frame, "HEAD", (cx - 18, cy + radius + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)
 
def draw_progress_bar(frame, pos_sec, total_sec, w, h):
    bar_x, bar_y = 10, h - 40
    bar_w = w - 20
    bar_h = 8
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
    if total_sec > 0:
        fill = int(bar_w * min(pos_sec / total_sec, 1.0))
        if fill > 0:
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h), (0, 200, 255), -1)
    elapsed = f"{int(pos_sec // 60):02d}:{int(pos_sec % 60):02d}"
    total   = f"{int(total_sec // 60):02d}:{int(total_sec % 60):02d}"
    cv2.putText(frame, f"{elapsed} / {total}", (bar_x, bar_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)
 
def draw_speed_bar(frame, speed, x, y, width=120, height=12):
    cv2.rectangle(frame, (x, y), (x + width, y + height), (40, 40, 40), -1)
    ratio = (speed - MIN_SPEED) / (MAX_SPEED - MIN_SPEED)
    fill  = int(np.clip(ratio, 0, 1) * width)
    color = (0, 255, 100) if 0.95 <= speed <= 1.05 else (0, 180, 255)
    if fill > 0:
        cv2.rectangle(frame, (x, y), (x + fill, y + height), color, -1)
    mid = int((1.0 - MIN_SPEED) / (MAX_SPEED - MIN_SPEED) * width)
    cv2.line(frame, (x + mid, y - 2), (x + mid, y + height + 2), (255, 255, 255), 1)
 
print("\nFaceEQ running — turn your head to control!")
print("SPACE=play/pause  R=reset  Q=quit\n")
 
while True:
    ret, frame = cap.read()
    if not ret:
        break
 
    frame    = cv2.flip(frame, 1)
    rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results  = face_mesh.process(rgb)
    face_detected = False
 
    if results.multi_face_landmarks:
        face_detected = True
        lm = results.multi_face_landmarks[0].landmark
 
        raw_yaw, raw_pitch = estimate_head_pose(lm, FRAME_W, FRAME_H)
        smooth_yaw   += SMOOTHING * (raw_yaw   - smooth_yaw)
        smooth_pitch += SMOOTHING * (raw_pitch - smooth_pitch)
        current_yaw   = smooth_yaw
        current_pitch = smooth_pitch
 
        if abs(current_yaw) > YAW_DEAD_ZONE and engine.is_playing:
            past_dz     = current_yaw - (YAW_DEAD_ZONE if current_yaw > 0 else -YAW_DEAD_ZONE)
            scrub_delta = (past_dz / 25.0) * SCRUB_SPEED
            engine.position_sec = engine.position_sec + scrub_delta
 
        if abs(current_pitch) > PITCH_DEAD_ZONE:
            past_dz      = current_pitch - (PITCH_DEAD_ZONE if current_pitch > 0 else -PITCH_DEAD_ZONE)
            speed_offset = (past_dz / 20.0) * (MAX_SPEED - 1.0)
            engine.speed = 1.0 + speed_offset
        else:
            engine.speed = 1.0
 
        nose = lm[NOSE_TIP]
        nx, ny = int(nose.x * FRAME_W), int(nose.y * FRAME_H)
        cv2.circle(frame, (nx, ny), 5, (0, 255, 255), -1)
 
    else:
        current_yaw   = 0.0
        current_pitch = 0.0
 
    pos_sec = engine.position_sec
    spd     = engine.speed
    playing = engine.is_playing
 
    status  = "PLAYING" if playing else "PAUSED"
    s_color = (0, 220, 60) if playing else (60, 60, 255)
    cv2.putText(frame, status, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, s_color, 2)
 
    if not face_detected:
        cv2.putText(frame, "No face — stay centered", (10, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 80, 255), 2)
 
    cv2.putText(frame, f"Yaw:   {current_yaw:+.1f}°",  (10, 65),  cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    cv2.putText(frame, f"Pitch: {current_pitch:+.1f}°", (10, 88),  cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    cv2.putText(frame, f"Speed: {spd:.2f}x",            (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255),   1)
    draw_speed_bar(frame, spd, 10, 122)
 
    if face_detected and abs(current_yaw) > YAW_DEAD_ZONE:
        direction = ">> FORWARD" if current_yaw > 0 else "<< REWIND"
        cv2.putText(frame, direction, (10, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 220), 2)
 
    draw_gauge(frame, FRAME_W - 75, 75, 55, current_yaw, current_pitch)
    draw_progress_bar(frame, pos_sec, TRACK_DURATION, FRAME_W, FRAME_H)
    cv2.putText(frame, "SPACE=play/pause  R=reset  Q=quit",
                (10, FRAME_H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (90, 90, 90), 1)
 
    cv2.imshow("FaceEQ — Day 05", frame)
 
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):
        playing = engine.toggle()
        print("▶ Playing" if playing else "⏸ Paused")
    elif key == ord('r'):
        engine.reset()
        engine.play()
        print("⏮ Reset → playing")
 
cap.release()
cv2.destroyAllWindows()
engine.stop()