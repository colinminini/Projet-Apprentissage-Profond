import torch


EXPERIMENTS = [
    {    
        "name": "full_30s",
        "metadata_path": "data/metadata_30_secs.csv",
        "clip_duration": 30.0,
        "segments_per_clip": 1,
        "batch_size": 4,
        "log_dir": "runs/logs/CNN_exp_full",
        "checkpoint_dir": "runs/checkpoints/CNN_exp_full",
        "num_epochs": 5,
        "device": torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    },

    {
        "name": "ensemble_3s",
        "metadata_path": "data/metadata_ensemble.csv",
        "clip_duration": 3.0,
        "segments_per_clip": 10,
        "batch_size": 16,
        "log_dir": "runs/logs/CNN_exp_ensemble",
        "checkpoint_dir": "runs/checkpoints/CNN_exp_ensemble",
        "num_epochs": 5,
        "device": torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    },

]
