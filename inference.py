import pandas as pd
import numpy as np
import os
import pickle
base_dir = "./"

# Loading the training_dataset object
with open(os.path.join(base_dir,"training_dataset.pkl"), 'rb') as f:
    training_dataset = pickle.load(f)

from pytorch_forecasting.metrics import MAE, QuantileLoss
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.data import TimeSeriesDataSet
import torch

# The parameters here won't matter because we rewrite them later
tft = TemporalFusionTransformer.from_dataset(
    training_dataset,
    learning_rate=5e-3,
    hidden_size=16,
    attention_head_size=2,
    dropout=0.2,
    hidden_continuous_size=8,
    loss=QuantileLoss(),
    log_interval=10,  # uncomment for learning rate finder and otherwise, e.g. to 10 for logging every 10 batches
    optimizer="Ranger",
    reduce_on_plateau_patience=4,
)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
best_model_path = os.path.join(base_dir,"epoch4-step5350.ckpt")
saved_state_dict = torch.load(best_model_path, map_location=device)
# Load the state dictionary into the TemporalFusionTransformer model
tft.load_state_dict(saved_state_dict['state_dict'])

## Create dataloader
data_set = pd.read_csv("test.csv")
test_dataset = TimeSeriesDataSet.from_dataset(training_dataset, data_test, stop_randomization=True)
max_batch_size = 32
batch_size=largest_even_divisor(len(test_dataset), max_batch_size) # this is a workaround when the number of samples is not dividable by batch_size
test_dataloader = test_dataset.to_dataloader(train=False, batch_size=batch_size, num_workers=4, drop_last=False)

# Predict values
predictions = tft.predict(test_dataloader, mode="prediction", 
                          return_y=True, 
                          trainer_kwargs=dict(accelerator=str(device), 
                                              enable_progress_bar=True))


    
