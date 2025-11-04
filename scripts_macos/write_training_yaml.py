# scripts_macos/write_training_yaml.py
import yaml, os
from pathlib import Path

config = {
  "window_step_ms": 10,
  "train_dir": "trained_models/wakeword",
  "features": [
    {"features_dir": "generated_augmented_features", "sampling_weight": 2.0, "penalty_weight": 1.0, "truth": True,  "truncation_strategy": "truncate_start", "type": "mmap"},
    {"features_dir": "negative_datasets/speech",     "sampling_weight": 12.0,"penalty_weight": 1.0, "truth": False, "truncation_strategy": "random",         "type": "mmap"},
    {"features_dir": "negative_datasets/dinner_party","sampling_weight": 12.0,"penalty_weight": 1.0,"truth": False,"truncation_strategy": "random",         "type": "mmap"},
    {"features_dir": "negative_datasets/no_speech",  "sampling_weight": 5.0, "penalty_weight": 1.0, "truth": False, "truncation_strategy": "random",         "type": "mmap"},
    {"features_dir": "negative_datasets/dinner_party_eval","sampling_weight": 0.0,"penalty_weight":1.0,"truth": False,"truncation_strategy":"split","type":"mmap"},
  ],
  "training_steps": [40000],
  "positive_class_weight": [1],
  "negative_class_weight": [20],
  "learning_rates": [0.001],
  "batch_size": 128,
  "time_mask_max_size": [0],
  "time_mask_count": [0],
  "freq_mask_max_size": [0],
  "freq_mask_count": [0],
  "eval_step_interval": 500,
  "clip_duration_ms": 1500,
  "target_minimization": 0.9,
  "minimization_metric": None,
  "maximization_metric": "average_viable_recall",
}

# Add personal features if they exist
if os.path.exists("personal_augmented_features/training"):
    config["features"].insert(1, {"features_dir": "personal_augmented_features", "sampling_weight": 3.0, "penalty_weight": 1.0, "truth": True, "truncation_strategy": "truncate_start", "type": "mmap"})
    print("✅ Added personal features with higher weight (3.0)")

with open("training_parameters.yaml", "w") as f:
    yaml.dump(config, f)
print("📝 Wrote training_parameters.yaml")