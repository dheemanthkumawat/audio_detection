import sounddevice as sd, numpy as np

# set your input device index here
sd.default.device = (1, None)
sd.default.samplerate = 16000
sd.default.channels   = 1

print("Recording 1 s…")
rec = sd.rec(int(1.0 * 16000), dtype='float32')
sd.wait()
print("Recorded shape:", rec.shape)
