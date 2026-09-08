"""Denoise a local WAV file with the saved ideal-ratio-mask model."""

from pathlib import Path
import torch
import librosa
import numpy as np
import soundfile as sf
from src.load_model import load_trained_model
from Config.paths import device, root_dir

model, params = load_trained_model()

SAMPLE_RATE = 16000
N_FFT = 512
HOP_LENGTH = 256

def denoise_real_audio(input_path, output_path):
    """Denoise a WAV file and write the result plus the normalized input."""
    print(f"\n--- Processing: {Path(input_path).name} ---")
    audio, sr = librosa.load(input_path, sr=SAMPLE_RATE)
    audio, _ = librosa.effects.trim(audio, top_db=20)
    audio = librosa.util.normalize(audio)
    original_length = len(audio)

    # Preserve phase because the model predicts a magnitude mask only.
    stft = librosa.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH)
    magnitude = np.abs(stft)
    phase = np.angle(stft)

    mag_log = np.log1p(magnitude).T
    mag_tensor = torch.tensor(mag_log, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        mask = model(mag_tensor).squeeze(0).cpu().numpy().T

    # Match training: apply the ratio mask to linear magnitudes.
    pred_mag = mask * magnitude

    stft_out = pred_mag * np.exp(1j * phase)
    denoised_audio = librosa.istft(stft_out, hop_length=HOP_LENGTH, n_fft=N_FFT, length=original_length)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, denoised_audio, SAMPLE_RATE)

    normalized_input = output_path.replace(".wav", "_normalized_input.wav")
    sf.write(normalized_input, audio, SAMPLE_RATE)

    print(f"Denoised file written to: {output_path}")

input_file = root_dir / "real test audios" / "wav files" / "real_clean_16k(4).wav"
output_file = root_dir / "real test audios" / "audio outputs" / "denoised_real_16k(4).wav"

denoise_real_audio(input_file, output_file)
