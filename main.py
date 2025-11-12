import numpy as np
from scipy.io.wavfile import write
import os

# Create the "audio" folder if it doesn't exist
os.makedirs("audio", exist_ok=True)

# Sampling rate (44.1 kHz)
rate = 44100

# Duration of each sound in seconds
duration = 2

# Helper function to create a tone
def make_tone(freq, filename):
    t = np.linspace(0, duration, int(rate * duration), endpoint=False)
    data = 0.5 * np.sin(2 * np.pi * freq * t)
    write(f"audio/{filename}", rate, data.astype(np.float32))
    print(f"✅ Created: audio/{filename}")

# Control samples — low tone
make_tone(220, "control1.wav")
make_tone(250, "control2.wav")

# Parkinson’s samples — higher tone
make_tone(440, "parkinson1.wav")
make_tone(480, "parkinson2.wav")
for i, freq in enumerate([220, 230, 240, 250, 260]):
    make_tone(freq, f"control{i+1}.wav")

for i, freq in enumerate([430, 440, 450, 460, 470]):
    make_tone(freq, f"parkinson{i+1}.wav")
