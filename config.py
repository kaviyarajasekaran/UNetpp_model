class Config:
    noisy_dir = "/kaggle/input/input-data2k/2kdata/Dataset-Ata100/Noisy"
    clean_dir = "/kaggle/input/input-data2k/2kdata/Dataset-Ata100/Clean"
    img_size = 384
    channels = 1
    batch_size = 1
    epochs = 50
    lr = 1e-4
    checkpoint_dir = "checkpoints"
    prediction_dir = "predictions"
