class Config:
    noisy_dir = "/kaggle/input/noisy"
    clean_dir = "/kaggle/input/clean"
    img_size = 384
    channels = 1
    batch_size = 1
    epochs = 50
    lr = 1e-4
    checkpoint_dir = "checkpoints"
    prediction_dir = "predictions"
