# live_demo.py

import queue, sounddevice as sd
import numpy as np
import tensorflow as tf, tensorflow_hub as hub
import librosa, csv
from vosk import Model as VoskModel, KaldiRecognizer
import json

# ─── CONFIG ────────────────────────────────────────────────────────────────
MIC_INDEX     = 24       # your USB mic index
DEVICE_SR     = 44100    # mic’s native sample rate
TARGET_SR     = 16000    # what YAMNet & Vosk expect
WINDOW_DUR    = 1.0      # seconds per analysis window
STRIDE_DUR    = 0.5      # seconds between inferences
SPEECH_THRESH = 0.3      # detection threshold for “Speech”

# ─── LOAD YAMNET ───────────────────────────────────────────────────────────
print("Loading YAMNet…")
yam       = hub.load("https://tfhub.dev/google/yamnet/1")
infer_yam = yam.signatures['serving_default']

csv_path    = yam.class_map_path().numpy().decode('utf-8')
class_names = [r['display_name'] for r in csv.DictReader(open(csv_path))]
speech_idx  = class_names.index("Speech")

# ─── LOAD VOSK STT ─────────────────────────────────────────────────────────
print("Loading Vosk model…")
vosk_model = VoskModel("models/vosk-small-en")      # ensure this path exists
recognizer = KaldiRecognizer(vosk_model, TARGET_SR)

# ─── SET UP AUDIO STREAM ────────────────────────────────────────────────────
sd.default.device     = (MIC_INDEX, None)
sd.default.samplerate = DEVICE_SR
sd.default.channels   = 1

q = queue.Queue()
def audio_cb(indata, frames, time, status):
    q.put(indata[:,0].copy())

stream = sd.InputStream(
    samplerate=DEVICE_SR,
    channels=1,
    blocksize=int(0.1 * DEVICE_SR),  # 100 ms
    callback=audio_cb
)
stream.start()

# ─── LIVE LOOP ──────────────────────────────────────────────────────────────
print("🔴 Live demo (Ctrl+C to quit) — printing top YAMNet class each second…")
buffer   = np.zeros((0,), dtype='float32')
win_size = int(WINDOW_DUR * DEVICE_SR)
stride   = int(STRIDE_DUR * DEVICE_SR)

while True:
    chunk  = q.get()
    buffer = np.concatenate([buffer, chunk])
    if len(buffer) < win_size:
        continue

    window = buffer[:win_size]
    buffer = buffer[stride:]

    # 1) resample → 16 kHz
    wav16 = librosa.resample(window,
                             orig_sr=DEVICE_SR,
                             target_sr=TARGET_SR)

    # 2) YAMNet inference
    out    = infer_yam(waveform=tf.constant(wav16, tf.float32))
    scores = out['output_0'].numpy().mean(axis=0)

    # 3) Print top-1 class for debugging
    top_i  = int(np.argmax(scores))
    print(f"[YAMNet] {class_names[top_i]} ({scores[top_i]:.3f})")

    # 4) If “Speech” → Vosk STT
    if scores[speech_idx] > SPEECH_THRESH:
        pcm = (wav16 * 32767).astype(np.int16).tobytes()
        if recognizer.AcceptWaveform(pcm):
            res = json.loads(recognizer.Result())
            txt = res.get("text", "").strip()
            if txt:
                print(f"🗣 Speech: {txt}")
