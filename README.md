# Jetson-Edge Audio Abnormality Detector  

Real-time edge pipeline that **listens to a USB microphone**, decides whether the current audio window contains *speech* or *other sounds*, and (soon) classifies those sounds as **normal vs abnormal**.  
The project is designed for NVIDIA Jetson boards so you can run everything locally without an internet connection.

---

## 1 . Current status

| Piece | Tech | Status |
|-------|------|--------|
| Audio capture | `sounddevice` | ✅ |
| Audio event classifier | **YAMNet** (TF-Hub) | ✅ |
| Speech-to-text | **Vosk-small-en** | ✅ (will be swapped for Whisper) |
| Abnormality classifier | — | 🚧 *(coming next)* |
| Deployment scripts | Jetson/x86 | 🚧 |

The live demo (`live_demo.py`) already shows:

```

\[YAMNet] Speech (0.86)
🗣 Speech: hello detector test
\[YAMNet] Music (0.71)
...

```

---

## 2 . Folder layout

```
audio_detection/
├── app/ # future REST / gRPC / websocket service code
├── data/ # sample & evaluation WAVs
├── logs/ # runtime logs (rotated daily)
├── models/
│ └── vosk-small-en/ # offline STT model (≈40 MB)
├── .venv/ # optional Python virtual-env (ignored by Git)
├── live_demo.py # YAMNet → Vosk proof-of-concept
├── miaow_16k.wav # tiny cat meow clip for sanity checks
├── detector.service # systemd unit to auto-start on boot
├── requirements.txt # pip dependencies
├── .gitignore
└── README.md # you’re reading it

````

*(Empty folders are intentional placeholders so CI/CD and Docker mounts don’t break once we add content.)*

---

## 3 . Quick-start (Jetson Orin / Xavier NX)

1. `sudo apt install python3-pip libsndfile1`
2. Clone & install deps  

   ```bash
   git clone https://github.com/your-org/jetson-audio-detector.git
   cd audio_detection
3. create a python venv using
   `python -m venv venv`
   `pip3 install -r requirements.txt` 

4. Plug in a **USB mic** and find its index:

   ```python
   python - <<'EOF'
   import sounddevice as sd, json
   print(json.dumps(sd.query_devices(), indent=2))
   EOF
   ```

5. Edit `live_demo.py` → set `MIC_INDEX` to that number.

6. Run the demo:

   ```bash
   python3 live_demo.py
   ```

   Press **Ctrl + C** to stop.

### Tweaking latency & accuracy

* `WINDOW_DUR` — shorter means faster reaction, but less context
* `SPEECH_THRESH` — raise in noisy rooms, lower in quiet booths
* Change `DEVICE_SR` to match your mic’s native rate to avoid resampling artefacts.

---

## 4 . Road-map / TODO

| Milestone             | Description                                                                                                                      |                                                       |
| --------------------- | -------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------- |
| **Whisper STT**       | Replace Vosk with `openai-whisper` (tiny-int8 on Jetson) for better accuracy in accents/noisy scenes.                            |                                                       |
| **Sound-type model**  | Train a small CNN/CRNN that flags *abnormal* acoustic events (alarms, glass breaks, shrieks) <br>→ `classifiers/abnormality.py`. |                                                       |
| **Modular CLI**       | `detector.py --mode speech --mode abnormal`, configurable YAML.                                                                  |                                                       |
| **Docker image**      | CUDA-enabled base for Jetson (*l4t-base-runtime + TF + PyTorch + FFmpeg*).                                                       |                                                       |
| **OTA update script** | Simple \`curl                                                                                                                    | sh\` to pull the latest model & scripts in the field. |
| **WebSocket service** | Publish JSON events to a local dashboard (Grafana/React).                                                                        |                                                       |

---

## 5 . Contributing

1. Fork → feature branch → PR (use conventional commits).
2. Commit messages: `feat`, `fix`, `docs`, `refactor`, `perf`, `test`.
3. Before PR, run `pytest` and `flake8`.

---

## 6 . License

MIT (see `LICENSE` file).

---

## 7 . Credits

* **YAMNet** – Google Research
* **Vosk** – Alpha Cephei

```
