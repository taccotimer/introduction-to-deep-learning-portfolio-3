from testing_framework import test_models
from training_framework import train_main_runs

if __name__ == "__main__":
    train_main_runs(
        config_files="config_files/config.json",
        trained_models_dir="trained_models",
        with_seed=True
        
    )

    test_models(
        config_file="config_files/training_config.json",
        trained_models_dir="trained_models",
        with_seed=True
    )
