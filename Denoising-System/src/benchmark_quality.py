import torch
import numpy as np
from src.metrics import compute_si_sdr
from Config.paths import device

def evaluate_perceptual_quality(model, valid_clean, valid_noisy):
    """
    Technique: Inverts predicted log-magnitudes back to linear space using expm1 and compares SI-SDR scores.
    Reason: Mathematically validates that the model actually suppressed background noise without degrading the target speech.
    """
    print("PERCEPTUAL AUDIO METRICS (VAL SET SUBSET)")

    if len(valid_clean) == 0:
        print("Warning: No validation samples found to evaluate.")
        return

    sisdr_before_list, sisdr_after_list = [], []
    num_eval_samples = min(200, len(valid_clean))

    with torch.no_grad():
        for i in range(num_eval_samples):
            cf, nf = valid_clean[i], valid_noisy[i]

            c_log = torch.tensor(np.load(cf).T, dtype=torch.float32).unsqueeze(0).to(device)
            n_log = torch.tensor(np.load(nf).T, dtype=torch.float32).unsqueeze(0).to(device)

            pred_log = model(n_log)
            pred_log_clamped = torch.clamp(pred_log, min=0.0)

            c_lin = torch.expm1(c_log).cpu().numpy().squeeze()
            n_lin = torch.expm1(n_log).cpu().numpy().squeeze()
            pred_lin = torch.expm1(pred_log_clamped).cpu().numpy().squeeze()

            s_b = compute_si_sdr(c_lin.flatten(), n_lin.flatten())
            s_a = compute_si_sdr(c_lin.flatten(), pred_lin.flatten())
            sisdr_before_list.append(s_b)
            sisdr_after_list.append(s_a)

    avg_sisdr_b = np.mean(sisdr_before_list)
    avg_sisdr_a = np.mean(sisdr_after_list)
    sisdr_imp = avg_sisdr_a - avg_sisdr_b

    print(f"SI-SDR Before : {avg_sisdr_b:.2f} dB")
    print(f"SI-SDR After : {avg_sisdr_a:.2f} dB")
    print(f"SI-SDR Improvement : {sisdr_imp:+.2f} dB")
    print("STOI Intelligibility : Not computed (pipeline saves magnitude only, no phase — invalid without it)")
    print("PESQ Speech Quality : Not computed (pipeline saves magnitude only, no phase — invalid without it)")