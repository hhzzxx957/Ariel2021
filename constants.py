import pathlib

__author__ = "Jason He"
__email__ = "hezhx@deepblueai.com"

project_dir = pathlib.Path(__file__).parent.absolute()

# paths to data dirs
params_train_path = project_dir / "data/params_train/home/ucapats/Scratch/ml_data_challenge/training_set/params_train"
lc_test_path = project_dir / "data/noisy_test/home/ucapats/Scratch/ml_data_challenge/test_set/noisy_test"
lc_train_path = project_dir / "data/noisy_train/home/ucapats/Scratch/ml_data_challenge/training_set/noisy_train"

data_test_path = project_dir / "data/full_test_signal_preprocessed.pt"
data_train_path = project_dir / "data/full_train_signal_preprocessed.pt" #full_train_signal_preprocessed.pt"

# fftdata_test_path = project_dir / "data/full_test_signal_fft.pt"
# fftdata_train_path = project_dir / "data/full_train_signal_fft.pt"

feat_train_path = project_dir / "data/train_properties_df.csv"
feat_test_path = project_dir / "data/test_properties_df.csv"
lgb_train_path = project_dir / "ml_method/lgb_oof_train_preds.txt"
lgb_test_path = project_dir / "ml_method/lgb_oof_test_preds.txt"
filefeat_train_path = project_dir / 'data/file_feat_train.pt'
filefeat_test_path = project_dir / 'data/file_feat_test.pt'

quantilefeat_train_path = project_dir / 'data/quantile_feat_train.npy'
quantilefeat_test_path = project_dir / 'data/quantile_feat_test.npy'

quantilephotonfeat_train_path = project_dir / 'data/quantile_featphoton_train.npy'
quantilephotonfeat_test_path = project_dir / 'data/quantile_featphoton_test.npy'

avg_val_path = project_dir / "data/average_signal_new.npy"

random_seed=2021

n_wavelengths = 55
n_timesteps = 300

val_size = 32
test_size = 32 #2048 #53900
train_size = 1256 - test_size # 125600 4096*4 #
batch_size = 1024 #int(train_size / 4)
lr = 1e-4

epochs = 120
save_from = 30
early_stop = 60

H1 = 1024
H2 = 256
