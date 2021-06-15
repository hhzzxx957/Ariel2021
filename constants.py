import pathlib
__author__ = "Jason He"
__email__ = "hezhx@deepblueai.com"


n_wavelengths = 55
n_timesteps = 300

train_size = 512
val_size = 512
epochs = 70
save_from = 20

H1 = 1024
H2 = 256

project_dir = pathlib.Path(__file__).parent.absolute()

# paths to data dirs
lc_train_path = project_dir / \
    "data/noisy_train/home/ucapats/Scratch/ml_data_challenge/training_set/noisy_train"
params_train_path = project_dir / \
    "data/params_train/home/ucapats/Scratch/ml_data_challenge/training_set/params_train"
