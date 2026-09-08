"""Convert a paired Hugging Face audio dataset into aligned spectrogram tensors."""

import io
import os
import soundfile as sf
import numpy as np
from datasets import load_dataset, Audio
from src.audio_utils import process_audio_array

def process_dataset_in_memory(download_path, final_clean_dir, final_noisy_dir):
    """Decode paired samples and save aligned log-spectrogram chunks as ``.npy``."""
    os.makedirs(final_clean_dir, exist_ok=True)
    os.makedirs(final_noisy_dir, exist_ok=True)

    print("Loading Hugging Face dataset......")
    dataset = load_dataset(download_path)
    dataset = dataset.cast_column("clean", Audio(decode=False))
    dataset = dataset.cast_column("noisy", Audio(decode=False))

    total = len(dataset["train"])
    total_chunks = 0
    skipped = 0

    for i, sample in enumerate(dataset["train"]):
        try:
            clean_bytes = sample["clean"]["bytes"]
            noisy_bytes = sample["noisy"]["bytes"]

            clean_array, sr_clean = sf.read(io.BytesIO(clean_bytes))
            noisy_array, sr_noisy = sf.read(io.BytesIO(noisy_bytes))

            clean_chunks = process_audio_array(clean_array, sr_clean)
            noisy_chunks = process_audio_array(noisy_array, sr_noisy)

            if len(clean_chunks) != len(noisy_chunks):
                skipped += 1
                continue

            base_name = f"sample{i:05d}"
            for chunk_idx, (c_spec, n_spec) in enumerate(zip(clean_chunks, noisy_chunks)):
                np.save(os.path.join(final_clean_dir, f"{base_name}_{chunk_idx}.npy"), c_spec)
                np.save(os.path.join(final_noisy_dir, f"{base_name}_{chunk_idx}.npy"), n_spec)
                total_chunks += 1

        except Exception as e:
            skipped += 1
            continue
        if i % 100 == 0:
            print(f"Processed {i}/{total} audio files. Generated {total_chunks} tensors so far.")

    print(f"\nExtraction & Processing Complete.")
    print(f"Total training chunks saved: {total_chunks}")
    print(f"Skipped {skipped} corrupted/mismatched files.")


process_dataset_in_memory(
    download_path="audio data",
    final_clean_dir=r"processed\cleaned data",
    final_noisy_dir=r"processed\noisy data"
    )
