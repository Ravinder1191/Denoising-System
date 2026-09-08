# How This Audio Denoising Project Works

This guide explains the engineering and signal-processing ideas in the project. It is separate from the setup README and is meant for understanding the code.

## The main idea

The project takes noisy speech, transforms it from a waveform into a time-frequency representation, predicts how much of each frequency bin should remain, and reconstructs a cleaner waveform.

```text
Phone microphone
    → IP Webcam HTTP audio stream
    → captured WAV bytes
    → waveform samples
    → STFT magnitude + phase
    → log magnitude frames
    → GRU predicts a mask
    → mask × original magnitude
    → original phase + inverse STFT
    → denoised WAV
```

The model does not generate speech from scratch. It predicts an **ideal ratio mask** (IRM): a value near `1` keeps a time-frequency region, while a value near `0` suppresses it.

## Why use an STFT?

Raw audio is a one-dimensional waveform. Noise suppression is easier in a representation where time and frequency are both visible.

For a 16 kHz signal, this project uses:

```python
N_FFT = 512
HOP_LENGTH = 256
```

`librosa.stft()` splits audio into overlapping 512-sample windows and converts each one into frequency bins. A real-valued 512-point FFT has 257 unique bins:

```text
number of frequency bins = N_FFT / 2 + 1 = 257
```

The model therefore receives tensors shaped like:

```text
(batch, time_frames, 257)
```

Each GRU step sees one audio frame across all 257 frequency bins. The GRU carries information across time, helping it distinguish short speech patterns from more stationary background noise.

## Magnitude and phase

The STFT is complex:

```python
stft = librosa.stft(audio)
magnitude = np.abs(stft)
phase = np.angle(stft)
```

It contains two different types of information:

- **Magnitude**: how strong each frequency is.
- **Phase**: timing alignment of the frequency components.

This model predicts a mask for the magnitude only. It keeps the original phase:

```python
denoised_magnitude = predicted_mask * magnitude
denoised_stft = denoised_magnitude * np.exp(1j * phase)
denoised_audio = librosa.istft(denoised_stft)
```

Using the noisy phase is common in a magnitude-mask baseline because phase prediction is substantially harder. It also explains why the output can still have some artifacts even when the magnitude denoising is effective.

## Why `log1p` and `expm1`?

Audio magnitudes have a large dynamic range. A few low-frequency or loud components can be much larger than quiet speech components.

```python
log_magnitude = np.log1p(magnitude)
```

`log1p(x)` is `log(1 + x)`. It compresses large values while remaining well behaved at zero. That makes the model input numerically easier to learn from.

Before applying a predicted mask, the script returns to linear magnitude:

```python
linear_magnitude = torch.expm1(model_input)
denoised_magnitude = mask * linear_magnitude
```

`expm1(x)` is `exp(x) - 1`, the inverse of `log1p(x)`.

The order matters. The mask is trained to act on linear magnitude, so this is correct:

```text
mask × expm1(log_magnitude)
```

This is not equivalent:

```text
expm1(mask × log_magnitude)
```

## The role of the GRU mask model

The model in `src/load_model.py` ends with a sigmoid:

```python
mask = sigmoid(linear(gru_output))
```

That constrains each mask value to `[0, 1]`.

```text
0.0  → remove this frequency component
0.5  → reduce it
1.0  → retain it
```

At inference time, the model receives a log-magnitude spectrogram and returns one mask value for every time frame and frequency bin. The model is loaded with `eval()` mode and run inside `torch.no_grad()` so dropout is disabled and gradients are not stored.

## Real-time file: what happens in order

`reat time.py` is not truly frame-by-frame real-time denoising yet. It is **record-then-process**:

1. Connect to the phone’s IP Webcam HTTP audio endpoint.
2. Keep receiving the live WAV response until you press Ctrl+C.
3. Decode the bytes into an audio array.
4. Save that captured phone-microphone recording as `original.wav`.
5. Run the whole recording through the model.
6. Save the output as `denoised.wav`.

The microphone belongs to the phone, not the laptop. The laptop only receives the stream over HTTP.

## Why `requests` is used

IP Webcam exposes audio through an HTTP URL, commonly:

```text
http://PHONE_IP:8080/audio.wav
```

`requests.get(..., stream=True)` opens that URL without downloading the entire response before code can start reading it:

```python
response = requests.get(URL, stream=True, timeout=(10, None))
```

This is necessary because live audio often has no natural end. The phone continues sending data until you stop the request or stop the camera server.

`response.raise_for_status()` checks for HTTP failures such as `404 Not Found` or `401 Unauthorized`. Without it, an HTML error page could be mistaken for audio bytes and later fail during decoding.

## Why audio arrives as bytes and chunks

Network data arrives as bytes, not NumPy arrays. A **byte** is an integer from 0 to 255; audio files are serialized sequences of these values.

```python
audio_bytes = bytearray()

for chunk in response.iter_content(chunk_size=4096):
    if chunk:
        audio_bytes.extend(chunk)
```

`iter_content()` yields pieces of the response as they arrive. A 4096-byte chunk is a practical network-buffer size; it is not an audio frame and has no special deep-learning meaning.

`bytearray` is used because it can be efficiently extended in place. Repeatedly doing `audio_bytes += chunk` creates new immutable `bytes` objects over and over, which becomes inefficient for longer recordings.

When you press Ctrl+C, the loop stops. The accumulated bytes now contain a WAV stream that can be decoded.

## Why `io.BytesIO` is used

`soundfile.read()` can read a filename or a file-like object. The received audio is in memory, not yet in a file, so `BytesIO` wraps it as an in-memory file:

```python
audio_array, source_rate = sf.read(io.BytesIO(bytes(audio_bytes)))
```

That avoids first writing a temporary WAV just to read it again.

## Why `soundfile` is used

`soundfile` reads and writes WAV data accurately and conveniently:

```python
sf.write(original_path, original, SAMPLE_RATE)
sf.write(denoised_path, denoised, SAMPLE_RATE)
```

The first write preserves the captured microphone signal. The second write stores the model output. Saving the original before inference is useful: even if the model throws an error or takes a long time, the recording itself is already safe.

## Why mono conversion and resampling are needed

The model was trained for mono, 16 kHz audio. A phone can send stereo or use another sample rate, so the script normalizes its input format:

```python
if audio_array.ndim > 1:
    audio_array = audio_array.mean(axis=1)

if source_rate != SAMPLE_RATE:
    audio_array = librosa.resample(
        audio_array.astype(float),
        orig_sr=source_rate,
        target_sr=16_000,
    )
```

Mixing channels produces mono. Resampling makes the waveform’s time scale match the training data; without it, frequencies and STFT frame timing would not mean the same thing to the model.

## Training pipeline

The project has two ways to form training data:

1. `scripts/building data.py` processes an already paired clean/noisy dataset.
2. `src/augmentation.py` creates mixtures dynamically from clean speech, noise recordings, and optional room impulse responses (RIRs).

The augmentation dataset does the following for every item:

```text
clean speech chunk
    → optional room impulse response convolution
    → random noise selection
    → mix at a random signal-to-noise ratio
    → shared random gain
    → peak normalization
    → log-magnitude STFT tensors
```

### SNR mixing

Signal-to-noise ratio is the relative power of clean speech and noise:

```text
SNR(dB) = 10 × log10(clean_power / noise_power)
```

The code measures the clean and noise powers, rescales the noise to the sampled target SNR, then adds it to the clean signal. Training across a range such as `-5 dB` to `15 dB` teaches the model to handle difficult and easy noise conditions.

### Room impulse responses

Convolving speech with a room impulse response simulates reflections and reverberation:

```python
reverbed = fftconvolve(audio, rir, mode="full")
```

This helps the model avoid overfitting to dry, studio-like speech.

## Validation and benchmarks

`src/load_data.py` splits validation data by original utterance name rather than by random chunks. This matters because chunks from the same source file are highly similar; putting some in training and some in validation would overstate model performance.

`src/benchmark_speed.py` computes **real-time factor**:

```text
RTF = processing time / audio duration
```

An RTF below 1 means the model can process audio faster than its duration. This measures model speed, but it is not the same as a full low-latency streaming system.

`src/metrics.py` computes SI-SDR. It is useful here because it measures separation quality while allowing overall volume scaling.

## What “real-time” would mean next

For genuine low-latency denoising, do not wait for Ctrl+C or collect the full stream. Instead:

```text
incoming audio chunk
    → keep a rolling STFT buffer
    → run a short frame or small batch through the GRU
    → overlap-add inverse STFT output
    → play or forward the result immediately
```

That needs careful buffering, overlap-add handling, GRU hidden-state management, and a latency budget. The current script is a correct offline test of a live source, not this continuous output pipeline.

## Useful debugging order

When no file is saved, check in this order:

1. Open `http://PHONE_IP:8080` in the laptop browser. The IP Webcam page must load.
2. Open `http://PHONE_IP:8080/audio.wav`. The browser or a media player should receive audio.
3. Press Ctrl+C in the script and wait for `Saved original recording:`.
4. Check `audio outputs/original.wav`. If it exists, capture worked.
5. Wait for `Saved denoised recording:` and check `audio outputs/denoised.wav`.

If step 4 works but step 5 does not, capture is fine and the issue is in model loading or inference. If step 4 does not work, focus on the phone URL, Wi-Fi/VPN connection, stream format, and output folder.
