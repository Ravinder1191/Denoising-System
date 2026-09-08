"""Export clean training WAV files for the augmentation notebook."""

from extract_waves import extract_clean_speech

extract_clean_speech(
    download_path="audio data",
    out_clean_dir="data/clean_raw_for_augment",
)
