import numpy as np
import librosa
import random
from scipy.signal import fftconvolve
import torch
from torch.utils.data import Dataset


class AugmentedDenoiseDataset(Dataset):
    """Create randomized noisy/clean log-spectrogram pairs from WAV files."""

    def __init__(self, clean_files, noise_files, rir_files,
                 sample_rate=16000, chunk_duration=2.0,
                 n_fft=512, hop_length=256,
                 snr_range=(-5, 15), gain_range_db=(-6, 6)):
        self.clean_files = clean_files
        self.noise_files = noise_files
        self.rir_files = rir_files
        self.sr = sample_rate
        self.chunk_samples = int(chunk_duration * sample_rate)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.snr_range = snr_range
        self.gain_range_db = gain_range_db

    def __len__(self):
        """Number of clean speech files available (one sample per epoch pass)."""
        return len(self.clean_files)

    def _load_chunk(self, filepath):
        """Load, resample, and crop or pad one waveform to the chunk length."""
        audio, _ = librosa.load(filepath, sr=self.sr)
        if len(audio) < self.chunk_samples:
            audio = np.pad(audio, (0, self.chunk_samples - len(audio)))
        else:
            start = random.randint(0, len(audio) - self.chunk_samples)
            audio = audio[start:start + self.chunk_samples]
        return audio

    def _apply_rir(self, audio):
        """Apply a random room impulse response without changing peak level."""
        rir_path = random.choice(self.rir_files)
        rir, _ = librosa.load(rir_path, sr=self.sr)
        reverbed = fftconvolve(audio, rir, mode="full")[:len(audio)]
        if np.abs(reverbed).max() > 0:
            reverbed = reverbed / np.abs(reverbed).max() * np.abs(audio).max()
        return reverbed

    def _mix_at_snr(self, clean, noise, snr_db):
        """Scale and add noise so the mixture reaches ``snr_db``."""
        clean_power = np.mean(clean ** 2) + 1e-8
        noise_power = np.mean(noise ** 2) + 1e-8
        target_noise_power = clean_power / (10 ** (snr_db / 10))
        noise_scaled = noise * np.sqrt(target_noise_power / noise_power)
        return clean + noise_scaled

    def __getitem__(self, idx):
        """Return an augmented noisy/clean pair with shape ``(frames, bins)``."""
        clean = self._load_chunk(self.clean_files[idx])
        noise = self._load_chunk(random.choice(self.noise_files))

        # Reverberation is optional so the model sees both dry and room speech.
        if random.random() < 0.5 and len(self.rir_files) > 0:
            clean = self._apply_rir(clean)

        snr_db = random.uniform(*self.snr_range)
        noisy = self._mix_at_snr(clean, noise, snr_db)

        # Shared gain preserves the selected SNR.
        gain_db = random.uniform(*self.gain_range_db)
        gain = 10 ** (gain_db / 20)
        clean = clean * gain
        noisy = noisy * gain

        # Keep STFT inputs within a consistent numeric range.
        peak = max(np.abs(noisy).max(), np.abs(clean).max(), 1e-8)
        clean = clean / peak
        noisy = noisy / peak

        clean_stft = librosa.stft(clean, n_fft=self.n_fft, hop_length=self.hop_length)
        noisy_stft = librosa.stft(noisy, n_fft=self.n_fft, hop_length=self.hop_length)

        clean_mag = np.log1p(np.abs(clean_stft)).T
        noisy_mag = np.log1p(np.abs(noisy_stft)).T

        return (torch.tensor(noisy_mag, dtype=torch.float32),
                torch.tensor(clean_mag, dtype=torch.float32))
