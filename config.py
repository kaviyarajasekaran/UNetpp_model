class Config:
    noisy_dir = "/kaggle/input/input-data2k/2kdata/Dataset-1k/New_Data100/Noisy"
    clean_dir = "/kaggle/input/input-data2k/2kdata/Dataset-1k/New_Data100/Clean"
    img_size = 384
    channels = 1
    batch_size = 2
    epochs = 30
    lr = 5e-5
    checkpoint_dir = "checkpoints"
    prediction_dir = "predictions"
    val_ratio = 0.2
    num_workers = 0
    pin_memory = True
