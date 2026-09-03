import requests
import soundfile as sf
import io

url = "http://<phone-ip>:8080/audio.wav"
r = requests.get(url, stream=True, timeout=10)

audio_bytes = b""
for chunk in r.iter_content(chunk_size=4096):
    audio_bytes += chunk
    if len(audio_bytes) > 16000 * 2 * 5:  # ~5 seconds at 16kHz, 16-bit
        break

audio_array, sr = sf.read(io.BytesIO(audio_bytes))
sf.write("outputs/original_noisy.wav", audio_array, sr)

#sf.write("outputs/denoised_output.wav", denoised_array, sr)