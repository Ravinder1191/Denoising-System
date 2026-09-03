from pathlib import Path
import torch

root_dir = Path(__file__).resolve().parent.parent

# Folder directories
model_dir = root_dir / "models"
output_dir = root_dir / "outputs"
data_dir = root_dir / "processed"

# Exact asset paths inside their respective folders
best_model = model_dir / "best_model_augmented.pth"
best_params_path = output_dir / "best_params_augmented.json"
clean_dir = data_dir / "cleaned data"
noisy_dir = data_dir / "noisy data"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")