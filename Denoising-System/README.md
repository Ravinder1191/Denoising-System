# Denoising System

A speech-denoising project built around a GRU that predicts an ideal ratio mask (IRM) for each short-time Fourier transform (STFT) frequency bin. The model uses the mask to attenuate noise while retaining the input phase for waveform reconstruction.

## What is included

- `src/augmentation.py` creates randomized clean/noisy spectrogram pairs, including variable SNR, gain, and optional room reverberation.
- `scripts/building data.py` converts a paired Hugging Face audio dataset into aligned `.npy` log-spectrogram chunks.
- `augmented training.ipynb` trains the augmented GRU model and saves its weights and hyperparameters.
- `raw audios.py` denoises a local WAV file.
- `reat time.py` fetches audio from an HTTP endpoint and writes the original and denoised WAV files.
- `src/test_model.py` reports inference latency and validation SI-SDR.

## Requirements

Use Python 3.10 or later and install the project dependencies:

```bash
pip install -r requirements.txt
```

The versions in `requirements.txt` are pinned for Python 3.10-3.12. CUDA is used automatically when the installed PyTorch build detects a compatible GPU; otherwise the project runs on CPU.

## Project structure and file guide

```text
Denoising-System/
├── Config/
│   └── paths.py                 Shared project paths, model-artifact names, and CPU/GPU selection.
├── models/
│   └── best_model_augmented.pth Saved GRU weights produced by training.
├── outputs/
│   ├── best_params_augmented.json  Hyperparameters required to rebuild the saved model.
│   ├── demo_original_noisy.wav     Example captured input audio.
│   └── demo_denoised_clean.wav     Example output from the streaming workflow.
├── real test audios/
│   ├── wav files/                Local WAV inputs for manual listening tests.
│   └── audio outputs/            Denoised WAV files created by the local inference script.
├── scripts/
│   ├── extract_waves.py          Exports clean WAV files from a paired Hugging Face dataset.
│   ├── building data.py          Converts paired clean/noisy audio into aligned training tensors.
│   └── building dataset.py       Small runner that exports clean speech for augmentation training.
├── src/
│   ├── __init__.py               Marks the source directory as an importable Python package.
│   ├── augmentation.py           Builds on-the-fly noisy/clean training pairs with SNR, gain, and RIR variation.
│   ├── audio_utils.py            Reusable waveform trimming, normalization, chunking, and STFT helpers.
│   ├── benchmark_quality.py      Calculates validation-set SI-SDR before and after denoising.
│   ├── benchmark_speed.py        Measures frame latency and real-time factor.
│   ├── load_data.py              Locates matching validation tensors while avoiding utterance-level leakage.
│   ├── load_model.py             Recreates the saved IRM model and loads its weights.
│   ├── metrics.py                Implements the scale-invariant SDR metric.
│   ├── model.py                  Contains the general GRU baseline architecture.
│   └── test_model.py             Runs the latency and quality benchmark workflow.
├── augmented training.ipynb      End-to-end augmented-model training notebook.
├── raw audios.py                 Denoises one local WAV file and saves a normalized input copy.
├── reat time.py                  Retrieves audio from an HTTP endpoint, then saves original and denoised files.
├── requirements.txt              Exact Python package versions needed to run the project and notebook.
└── README.md                     This setup and usage guide.
```
`processed/` is created when preprocessing runs. It is intentionally not included above because it can become large: it contains `cleaned data/` and `noisy data/` directories of matching `.npy` log-spectrogram chunks.

The `.idea/` directory is created by PyCharm for local editor settings. `__pycache__/` directories are generated automatically by Python and can be ignored.

## Prepare training data

1. Set the dataset identifier and output locations in `scripts/building data.py`.
2. Run the script. It reads paired `clean` and `noisy` audio columns, resamples audio to 16 kHz, trims silence, normalizes it, splits it into two-second chunks, and saves matching log-magnitude STFT tensors.
3. If you want to train with augmentation, use `scripts/extract_waves.py` to export clean speech, then set the clean-speech, noise, and RIR paths in `augmented training.ipynb`.

The preprocessing format is important: each tensor has 257 frequency bins because the default STFT uses `n_fft=512`.

## Train

Open and run `augmented training.ipynb`. The notebook trains on clean speech mixed with random noise and saves:

- `models/best_model_augmented.pth`
- `outputs/best_params_augmented.json`

Keep these filenames aligned with `Config/paths.py`, or update the paths there.

## Denoise a WAV file

Update `input_file` and `output_file` in `raw audios.py`, then run:

```bash
python "raw audios.py"
```

The script writes the denoised WAV and a normalized copy of the input. It uses the original STFT phase, so it improves magnitude content only; phase artifacts can remain in difficult recordings.

## Evaluate

After preparing `processed/cleaned data` and `processed/noisy data`, run:

```bash
python -m src.test_model
```

This reports single-frame inference latency and SI-SDR for a validation subset. STOI and PESQ are intentionally not reported because the stored dataset contains magnitudes only, not reconstructable phase-aware waveforms.

## Notes

- Input audio is converted to mono by Librosa and resampled to 16 kHz.
- The validation split is based on source utterance names to reduce chunk-level data leakage.
- Paths in the scripts and notebook are examples and must be updated for the machine running the project.
