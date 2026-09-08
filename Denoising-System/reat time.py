"""Capture audio from IP Webcam and save original and denoised WAV files."""
import os
import io
import librosa
import numpy as np
import requests
import soundfile as sf
import torch
from Config.paths import device
from src.load_model import load_trained_model


SAMPLE_RATE = 16_000
N_FFT = 512
HOP_LENGTH = 256
URL = "http://192.168.1.4:8080/audio.wav"

model, _ = load_trained_model()

def denoise_array(audio):
    """Apply the trained mask model to a mono audio waveform."""
    stft = librosa.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH)
    magnitude = np.abs(stft)
    phase = np.angle(stft)

    log_magnitude = np.log1p(magnitude).T
    model_input = torch.tensor(log_magnitude, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        mask = model(model_input)

    linear_magnitude = torch.expm1(model_input)
    denoised_magnitude = (mask * linear_magnitude).squeeze(0).cpu().numpy().T
    denoised_stft = denoised_magnitude * np.exp(1j * phase)
    return librosa.istft(denoised_stft, hop_length=HOP_LENGTH, n_fft=N_FFT)


print("Recording started. Press Ctrl+C to stop.")

response = requests.get(URL, stream=True, timeout=(10, None))
response.raise_for_status()
audio_bytes = bytearray()

try:
    for chunk in response.iter_content(chunk_size=4096):
        if chunk:
            audio_bytes.extend(chunk)
except KeyboardInterrupt:
    print("\nStopped. Processing recording...")
finally:
    response.close()

if not audio_bytes:
    raise RuntimeError("IP Webcam returned no audio data.")

audio_array, source_rate = sf.read(io.BytesIO(bytes(audio_bytes)))
if audio_array.ndim > 1:
    audio_array = audio_array.mean(axis=1)
if source_rate != SAMPLE_RATE:
    audio_array = librosa.resample(
        audio_array.astype(float),
        orig_sr=source_rate,
        target_sr=SAMPLE_RATE,
    )

original = audio_array.astype(np.float32)

source_path = r'C:\Users\Ravin\PycharmProjects\Denoising-System\audio outputs'
original_file = 'original.wav'
denoised_file = 'denoised.wav'

original_wav_file = os.path.join(source_path, original_file)
denoised_wav_file = os.path.join(source_path, denoised_file)

sf.write(original_wav_file, original, SAMPLE_RATE)
print(f"Saved original recording: {original_wav_file}")

print("Denoising...")
denoised = denoise_array(original)

sf.write(denoised_wav_file, denoised, SAMPLE_RATE)
print(f"Saved denoised recording: {denoised_wav_file}")
