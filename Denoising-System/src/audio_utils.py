import numpy as np
import librosa

def trim_silence(audio, top_db=20):
    """
    Technique: Removes dead silence from the start and end of the 1D audio array.
    Reason: Forces the network to learn actual speech and noise patterns instead of wasting compute on empty sound.
    """
    trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
    return trimmed

def normalize_audio(audio):
    """
    Technique: Scales the audio wave so its highest peak hits 1.0 or -1.0.
    Reason: Standardizes volume across all files, keeping training gradients stable so the model doesn't crash early.
    """
    return librosa.util.normalize(audio)

def segment_into_chunks(audio, chunk_duration=2.0, sr=16000):
    """
    Technique: Chops continuous audio into fixed 2-second blocks, padding the last block with zeros if it's too short.
    Reason: Recurrent networks (GRUs) need consistent input shapes to efficiently batch process matrices.
    """
    chunk_samples = int(chunk_duration * sr)
    chunks = []
    for start in range(0, len(audio), chunk_samples):
        chunk = audio[start:start + chunk_samples]
        if len(chunk) < chunk_samples:
            chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
        chunks.append(chunk)
    return chunks

def audio_to_log_spectrogram(audio_chunk, n_fft=512, hop_length=256):
    """
    Technique: Computes the STFT to get 257 frequency bins, takes the magnitude, and immediately applies log1p.
    Reason: Combines frequency isolation and logarithmic compression in RAM, avoiding giant intermediate file saves.
    """
    stft_result = librosa.stft(audio_chunk, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft_result)
    return np.log1p(magnitude)

def process_audio_array(audio, sr=16000):
    """
    Technique: Runs the full pipeline (trim, normalize, chunk, transform) directly on the loaded audio array.
    Reason: Keeps massive audio data in RAM only for milliseconds before it becomes lightweight .npy tensors.
    """
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    audio = trim_silence(audio)
    audio = normalize_audio(audio)
    chunks = segment_into_chunks(audio)
    log_spectrograms = [audio_to_log_spectrogram(c) for c in chunks]
    return log_spectrograms
