class Config:
    noisy_dir = "/kaggle/input/input-data2k/2kdata/Dataset-1k/New_Data100/Noisy"
    clean_dir = "/kaggle/input/input-data2k/2kdata/Dataset-1k/New_Data100/Clean"
    img_size = 384
    channels = 1
    batch_size = 1
    epochs = 30
    lr = 1e-4
    checkpoint_dir = "checkpoints"
    prediction_dir = "predictions"
    val_ratio = 0.2
    num_workers = 2
    pin_memory = True
