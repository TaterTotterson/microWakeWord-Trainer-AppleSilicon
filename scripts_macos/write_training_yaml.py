# scripts_macos/write_training_yaml.py
import yaml, os

generated_positive_sampling_weight = float(os.environ.get("MWW_GENERATED_POSITIVE_SAMPLING_WEIGHT", "2.0"))
personal_positive_sampling_weight = float(os.environ.get("MWW_PERSONAL_POSITIVE_SAMPLING_WEIGHT", "10.0"))
reviewed_negative_sampling_weight = float(os.environ.get("MWW_REVIEWED_NEGATIVE_SAMPLING_WEIGHT", "30.0"))
reviewed_negative_penalty_weight = float(os.environ.get("MWW_REVIEWED_NEGATIVE_PENALTY_WEIGHT", "4.0"))
reviewed_negative_truncation = os.environ.get("MWW_REVIEWED_NEGATIVE_TRUNCATION", "truncate_start")

config = {
  "window_step_ms": 10,
  "train_dir": "trained_models/wakeword",
  "features": [
    {"features_dir": "generated_augmented_features", "sampling_weight": generated_positive_sampling_weight, "penalty_weight": 1.0, "truth": True,  "truncation_strategy": "truncate_start", "type": "mmap"},
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
    config["features"].insert(1, {
        "features_dir": "personal_augmented_features",
        "sampling_weight": personal_positive_sampling_weight,
        "penalty_weight": 1.0,
        "truth": True,
        "truncation_strategy": "truncate_start",
        "type": "mmap",
    })
    print(
        "✅ Added personal features with higher weight "
        f"({personal_positive_sampling_weight})"
    )

# Add reviewed false-positive features if they exist
if os.path.exists("reviewed_negative_features/training"):
    insert_at = 2 if os.path.exists("personal_augmented_features/training") else 1
    config["features"].insert(insert_at, {
        "features_dir": "reviewed_negative_features",
        "sampling_weight": reviewed_negative_sampling_weight,
        "penalty_weight": reviewed_negative_penalty_weight,
        "truth": False,
        "truncation_strategy": reviewed_negative_truncation,
        "type": "mmap",
    })
    print(
        "✅ Added reviewed negative features with hard-negative weighting "
        f"(sampling={reviewed_negative_sampling_weight}, "
        f"penalty={reviewed_negative_penalty_weight}, "
        f"truncation={reviewed_negative_truncation})"
    )

with open("training_parameters.yaml", "w") as f:
    yaml.dump(config, f)
print("📝 Wrote training_parameters.yaml")
