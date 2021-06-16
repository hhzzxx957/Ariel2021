import pathlib

__author__ = "Jason He"
__email__ = "hezhx@deepblueai.com"

project_dir = pathlib.Path(__file__).parent.absolute()

# paths to data dirs
params_train_path = project_dir / "data/params_train/home/ucapats/Scratch/ml_data_challenge/training_set/params_train"
lc_test_path = project_dir / "data/noisy_test/home/ucapats/Scratch/ml_data_challenge/test_set/noisy_test"
lc_train_path = project_dir / "data/noisy_train/home/ucapats/Scratch/ml_data_challenge/training_set/noisy_train"

random_seed=2021

n_wavelengths = 55
n_timesteps = 300

train_size = 4096
val_size = 1024
test_size=1024
batch_size = 256 #int(train_size / 4)

epochs = 120
save_from = 20

H1 = 1024
H2 = 256

device_id = 2