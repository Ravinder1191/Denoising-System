import os
import glob
from Config.paths import clean_dir, noisy_dir

def load_validation_pairs():
    """
    Technique: Splits train/val data by unique file base names (utterances) instead of random individual chunks.
    Reason: Prevents data leakage where the same speaker's background noise appears in both training and testing phases.
    """
    all_clean_files = sorted(glob.glob(os.path.join(clean_dir, "*.npy")))
    base_names = sorted(set(os.path.basename(f).rsplit('_', 1)[0] for f in all_clean_files))
    train_size = int(0.85 * len(base_names))
    val_bases = set(base_names[train_size:])

    val_clean = [f for f in all_clean_files if os.path.basename(f).rsplit('_', 1)[0] in val_bases]
    val_noisy = [f.replace(str(clean_dir), str(noisy_dir)) for f in val_clean]

    valid_clean, valid_noisy = [], []
    for cf, nf in zip(val_clean, val_noisy):
        if os.path.exists(nf):
            valid_clean.append(cf)
            valid_noisy.append(nf)

    return valid_clean, valid_noisy