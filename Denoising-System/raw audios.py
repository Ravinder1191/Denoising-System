import os
import torch
import librosa
import numpy as np
import soundfile as sf
from src.load_model import load_trained_model
from Config.paths import device

# Load IRM model
model, params = load_trained_model()

SAMPLE_RATE = 16000
N_FFT = 512
HOP_LENGTH = 256

def denoise_real_audio(input_path, output_path):
    print(f"\n--- Processing: {os.path.basename(input_path)} ---")
    audio, sr = librosa.load(input_path, sr=SAMPLE_RATE)
    audio, _ = librosa.effects.trim(audio, top_db=20)
    audio = librosa.util.normalize(audio)
    original_length = len(audio)

    # 1. STFT
    stft = librosa.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH)
    magnitude = np.abs(stft)
    phase = np.angle(stft)

    # 2. Input log-spectrogram to get the IRM mask
    mag_log = np.log1p(magnitude).T
    mag_tensor = torch.tensor(mag_log, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        # Model outputs mask between [0.0, 1.0]
        mask = model(mag_tensor).squeeze(0).cpu().numpy().T

    # 3. Apply IRM in log-space matching training objective
    pred_log_clean = mask * mag_log.T
    pred_mag = np.expm1(pred_log_clean)

    # 4. Reconstruct with original phase
    stft_out = pred_mag * np.exp(1j * phase)
    denoised_audio = librosa.istft(stft_out, hop_length=HOP_LENGTH, n_fft=N_FFT, length=original_length)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sf.write(output_path, denoised_audio, SAMPLE_RATE)

    normalized_input = output_path.replace(".wav", "_normalized_input.wav")
    sf.write(normalized_input, audio, SAMPLE_RATE)

    print(f"Denoised file written to: {output_path}")

# calling the denoise audio function
input_file = r'C:\Users\Ravin\PycharmProjects\vokie\real test audios\wav files\real_clean_16k(4).wav'
output_file = r"C:\Users\Ravin\PycharmProjects\vokie\real test audios\audio outputs\denoised_real_16k(4).wav"

denoise_real_audio(input_file, output_file)