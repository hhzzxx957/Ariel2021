import pathlib

__author__ = "Jason He"
__email__ = "hezhx@deepblueai.com"

project_dir = pathlib.Path(__file__).parent.absolute()

# paths to data dirs
params_train_path = project_dir / "data/params_train/home/ucapats/Scratch/ml_data_challenge/training_set/params_train"
lc_test_path = project_dir / "data/noisy_test/home/ucapats/Scratch/ml_data_challenge/test_set/noisy_test"
lc_train_path = project_dir / "data/noisy_train/home/ucapats/Scratch/ml_data_challenge/training_set/noisy_train"

data_test_path = project_dir / "data/full_test_signal_fftsmooth.pt"
data_train_path = project_dir / "data/full_train_signal_fftsmooth.pt" #full_train_signal_preprocessed.pt"
feat_train_path = project_dir / "data/properties_df.csv"
feat_test_path = project_dir / "data/test_properties_df.csv"
lgb_train_path = project_dir / "ml_method/lgb_train_preds.txt"
lgb_test_path = project_dir / "ml_method/lgb_test_preds.txt"
file_train_path = project_dir / 'data/file_feat_train.pt'
file_test_path = project_dir / 'data/file_feat_test.pt'

avg_val_path = project_dir / "data/average_signal_new.npy"

random_seed=2021

n_wavelengths = 55
n_timesteps = 300

val_size = 2048
test_size = 2048 #2048 #53900
train_size = 125600 - test_size # 125600 4096*4 #
batch_size = 512 #int(train_size / 4)
lr = 1e-4

epochs = 150
save_from = 50
early_stop = 100

H1 = 1024
H2 = 256
