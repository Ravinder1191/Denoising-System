import time
import torch
import numpy as np
from Config.paths import device

def benchmark_latency(model):
    """
    Technique: Simulates live audio processing frame by frame.
    Reason: Accurately isolates GPU execution time to prove the Real-Time Factor (RTF) is safely below 1.0.
    """
    print("REAL-TIME LATENCY & SPEED BENCHMARK")

    dummy_frame = torch.randn(1, 1, 257).to(device)
    for _ in range(20):  # Warmup runs
        _ = model(dummy_frame)

    frame_times = []
    with torch.no_grad():
        for _ in range(200):
            t0 = time.perf_counter()
            _ = model(dummy_frame)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            frame_times.append((t1 - t0) * 1000)

    avg_frame_latency_ms = np.mean(frame_times)
    frame_duration_ms = (128 / 16000) * 1000
    rtf = avg_frame_latency_ms / frame_duration_ms

    print(f"Hardware Device  {device.type.upper()}")
    print(f"Inference Latency / Frame : {avg_frame_latency_ms:.3f} ms")
    print(f"STFT Frame Duration : {frame_duration_ms:.2f} ms")
    print(f"Real-Time Factor (RTF) : {rtf:.4f}")

    if rtf < 1.0:
        print(f"Speed Status : EXCELLENT (Model runs {1.0/rtf:.1f}x faster than real-time)")
    else:
        print(f"Speed Status : TOO SLOW FOR REAL-TIME (RTF must be < 1.0)")