from pytorch_forecasting.data import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import GroupNormalizer
from pytorch_forecasting.data.encoders import NaNLabelEncoder
import pandas as pd
import json
import config_utils

print("Reading data...")
data = pd.read_csv("train_data.csv")
# Load the configuration from the file
config_file_path = "./config/timeseriesdataset_config.json"

with open(config_file_path, 'r') as file:
    data_config = json.load(file)

# Load config
with open('./config/training_config.json') as config_file:
    training_config = json.load(config_file)

# Converting categorical features to category type
categorical_features = config_utils.get_categorical_variables(data_config)
for cat_feature in categorical_features:
    data[cat_feature] = data[cat_feature].astype(str).astype("category")

# Setting up the TimeSeriesDataSet
max_prediction_length = data_config['max_prediction_length']
max_encoder_length = data_config['max_encoder_length']
#training_cutoff = data["time_idx"].max() - 1000 # Uncomment if manually set

# Calculate the range of 'time_idx'
time_idx_range = data["time_idx"].max() - data["time_idx"].min()
# Calculate the cutoff as 90% of the range (to keep 10% for validation/testing)
training_cutoff = data["time_idx"].min() + int(time_idx_range * 0.8)

# When there is many targets we create a MultiNormalizer, otherwise just one
if len(data_config['target']) > 1:
    from pytorch_forecasting.data import MultiNormalizer
    target_normalizer = MultiNormalizer(
        [GroupNormalizer(groups=data_config['target_normalizer'], transformation="softplus") for _ in data_config['target']]
    )
else:
    target_normalizer = GroupNormalizer(groups=data_config['target_normalizer'], transformation="softplus")

print("Creating timeseriesdataset object...")
training = TimeSeriesDataSet(
    data[lambda x: x.time_idx <= training_cutoff],
    time_idx="time_idx",
    target=data_config["target"] if len(data_config["target"]) > 1 else data_config["target"][0],
    group_ids=data_config['group_ids'],
    min_encoder_length=data_config["min_encoder_length"],
    max_encoder_length=max_encoder_length,
    min_prediction_length=data_config["min_prediction_length"],
    max_prediction_length=max_prediction_length,
    static_categoricals=data_config["static_categoricals"],
    static_reals=data_config["static_reals"],
    time_varying_known_categoricals=data_config["time_varying_known_categoricals"],
    time_varying_known_reals=data_config['time_varying_known_reals'],
    time_varying_unknown_categoricals=data_config["time_varying_unknown_categoricals"],
    time_varying_unknown_reals=data_config["time_varying_unknown_reals"],
    target_normalizer=target_normalizer,
    add_relative_time_idx=data_config['add_relative_time_idx'],
    add_target_scales=data_config["add_target_scales"],
    add_encoder_length=data_config["add_encoder_length"],
    allow_missing_timesteps=data_config["allow_missing_timesteps"],
    categorical_encoders={
        col:NaNLabelEncoder(add_nan=True) for col in data_config['group_ids']
    }
)

import pickle 

with open("./training_dataset.pkl", 'wb') as f:
    pickle.dump(training, f)

# Create validation set
validation = TimeSeriesDataSet.from_dataset(training, data, predict=True, min_prediction_idx=training.index.time.max() + 1, stop_randomization=True)


batch_size=training_config['data_loader']['batch_size']
num_workers = training_config['data_loader']['num_workers']
# Set up dataloaders with the calculated number of workers
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=num_workers)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=num_workers)

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import MAE, QuantileLoss
from pytorch_lightning import loggers as pl_loggers
import torch
tensorboard = pl_loggers.TensorBoardLogger('./')
pl.seed_everything(42)

print("Training ...")
# Callbacks
early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=3, verbose=False, mode="min")
lr_logger = LearningRateMonitor()

# Determine device
device = 'gpu' if torch.cuda.is_available() else 'cpu'

# Trainer configuration
trainer_config = training_config["trainer"]
trainer_config.update({"callbacks": [lr_logger, early_stop_callback]})
trainer = pl.Trainer(**trainer_config)

# TFT configuration
tft_config = training_config["tft"]
tft = TemporalFusionTransformer.from_dataset(
    training,
    loss=QuantileLoss(),
    log_interval=10,
    **tft_config
)

print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")

# Fit network
trainer.fit(
    tft,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)

############ Evaluate performance ##########

# load the best model according to the validation loss
# (given that we use early stopping, this is not necessarily the last epoch)
best_model_path = trainer.checkpoint_callback.best_model_path
print("Model path : ", best_model_path)
best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

# calcualte mean absolute error on validation set
predictions = best_tft.predict(val_dataloader, return_y=True, trainer_kwargs=dict(accelerator=device))
print("MAE : ", MAE()(predictions.output, predictions.y))
