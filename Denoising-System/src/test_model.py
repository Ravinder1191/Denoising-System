from src.load_model import load_trained_model
from src.load_data import load_validation_pairs
from src.benchmark_speed import benchmark_latency
from src.benchmark_quality import evaluate_perceptual_quality
# setting up models and data
model, best_params = load_trained_model()
valid_clean, valid_noisy = load_validation_pairs()

benchmark_latency(model)
evaluate_perceptual_quality(model, valid_clean, valid_noisy)
