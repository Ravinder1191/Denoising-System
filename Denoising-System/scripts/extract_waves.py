"""Extract clean WAV files from a Hugging Face paired-audio dataset."""

import io
import os
import soundfile as sf
from datasets import load_dataset, Audio

def extract_clean_speech(download_path, out_clean_dir):
    """Write each clean audio sample from ``download_path`` to ``out_clean_dir``."""
    dataset = load_dataset(download_path)
    dataset = dataset.cast_column("clean", Audio(decode=False))
    dataset = dataset.cast_column("noisy", Audio(decode=False))

    total = len(dataset["train"])
    os.makedirs(out_clean_dir, exist_ok=True)

    for i, sample in enumerate(dataset["train"]):
        clean_bytes = sample["clean"]["bytes"]
        clean_array, sr = sf.read(io.BytesIO(clean_bytes))
        sf.write(os.path.join(out_clean_dir, f"sample{i:05d}.wav"), clean_array, sr)

        if i % 100 == 0:
            print(f"Processed {i}/{total} clean speech files.")
